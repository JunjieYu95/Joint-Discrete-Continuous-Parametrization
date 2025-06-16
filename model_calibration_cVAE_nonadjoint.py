
import numpy as np
from utils.load_model import load
from utils.dataloaders import get_straight_geological_scenario_dataloaders
import torch
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize, Bounds
import pickle
import matlab.engine
import matplotlib.pyplot as plt
import os
import shutil
from two_phase_flow_viz import plot_history_matching, plot_saturation_pressure

def one_hot_penalty(z,alpha, disc_size = 4):
    z = torch.tensor(z,dtype = torch.float32)
    penalty = torch.sum(z[-disc_size:]*(1-z[-disc_size:])).detach().numpy()
    return alpha*penalty

def one_hot_penalty_gradient(z,alpha, disc_size = 4):
    z = torch.tensor(z,dtype = torch.float32, requires_grad=True)
    penalty = torch.sum(z[-disc_size:]*(1-z[-disc_size:]))
    penalty.backward()
    penalty_grad = z.grad.detach().numpy()
    return alpha*penalty_grad

def two_phase_ref(xnorm, eng, MRST_folder):
    '''
    this function is used to generate state_ref file for later use, the input is normlized permeability'''
    # for reference case, we do not need to decode z, we directly accept xnorm
    xnorm = matlab.double(np.array(xnorm.flatten(), dtype = np.double).tolist())
    eng.two_phase_forward_ref(xnorm, MRST_folder)
    return 0

def two_phase_mismatch(z, eng, MRST_folder, parametrization_model):
    '''
    this function is used to get observation mismatch compared to reference case, 
    the input is parametrized perm field
    '''
    # decode z to get xnorm
    z = torch.tensor(z,dtype = torch.float32)
    xnorm = parametrization_model.decode(z).view(1, -1)
    xnorm_matlab = matlab.double(xnorm.flatten().tolist())
    mismatch = eng.two_phase_forward(xnorm_matlab, MRST_folder)
    return mismatch

def two_phase_mismatch_batch(z_batch, eng, MRST_folder, parametrization_model):
    '''
    this function is used to get observation mismatch given a batch of latent code z
    compared to reference case, 
    the input is parametrized perm field
    '''
    z_batch = torch.tensor(z_batch,dtype = torch.float32)
    batch_size = z_batch.shape[0]
    xnorm_batch = parametrization_model.decode(z_batch).view(batch_size, -1)
    xnorm_matlab_batch = matlab.double(xnorm_batch.tolist())
    mismatch_batch = eng.two_phase_forward_batch(xnorm_matlab_batch, MRST_folder)
    return mismatch_batch

def two_phase_mismatch_gradient(z, eng, MRST_folder, parametrization_model):
    '''
    this function is used to get observation mismatch gradient compared to reference case, 
    the input is parametrized perm field'''
    z = torch.tensor(z,dtype = torch.float32, requires_grad=True)
    xnorm = parametrization_model.decode(z).view(-1,1)
    # creat xnorm_matlab to avoid destroying xnorm's shape which will be used when calculating jaobian while decoding
    xnorm_matlab = matlab.double(xnorm.flatten().tolist())

    [mismatch, ad] = eng.two_phase_forward_with_adjoint(xnorm_matlab, MRST_folder,nargout=2)
    mismatch = np.array(mismatch)
    ad = np.array(ad)
    d_mis_d_x = ad.reshape(1,-1)

    ## get gradiant via chain rule and unit transform
    # get d_xnorm_d_z
    d_xnorm_d_z = torch.zeros(xnorm.numel(), z.numel())
    for i in range(xnorm.shape[0]):
        grad = torch.autograd.grad(xnorm[i], z, retain_graph=True)[0]
        d_xnorm_d_z[i] = grad
    
    # detach d_xnorm_d_z and transfer to numpy
    d_xnorm_d_z = d_xnorm_d_z.numpy()
    # print('d_xnorm_d_z.shape, should be [4096,14]',d_xnorm_d_z.shape)

    # unit transform
    mD = 9.8692e-16
    x_min = 10*mD
    x_max = 200*mD

    # detach xnorm and transfer to numpy
    xnorm = xnorm.detach().numpy()

    x = np.exp(xnorm*(np.log(x_max)-np.log(x_min))+np.log(x_min))
    d_x_dxnorm = (np.log(x_max)-np.log(x_min))*x.reshape(1,-1)

    d_mis_d_z =  np.matmul((d_mis_d_x*d_x_dxnorm),d_xnorm_d_z)
    return d_mis_d_z

def calibrator(ref_perm_latent, init_perm_latent, parametrization_model, obj_handle, gradient_handle, bounds, constraints, tol=1e-6, maxiter=1000, disp=False):
    """
    Calibrator
    """
    # get reference state
    ref_perm = parametrization_model.decode(ref_perm_latent).view(1, -1).flatten()
    # generate state_ref
    state_ref = obj_handle(ref_perm)

    # get initial state
    init_perm = parametrization_model.decode(init_perm_latent).view(1, -1).flatten()

    # optimiztaion
    res = minimize(obj_handle, init_perm, method='trust-constr', 
                   jac=gradient_handle, hess='3-point', bounds=bounds, constraints=constraints,
     tol=tol, options={'maxiter': maxiter, 'disp': disp})

    # get calibrated latent vector
    cal_perm_latent = res.x

    return cal_perm_latent

def tester(par_iter, opt_process_folder = 'opt_process_test_9spots_gamma10', monitor_interval = 1):

    # create a folder to save the optimization results for each iteration
    opt_process_folder = opt_process_folder + '_case_' + str(par_iter)
    if os.path.exists(opt_process_folder):
        shutil.rmtree(opt_process_folder)   
    os.mkdir(opt_process_folder)

    eng = matlab.engine.start_matlab()
    MRST_folder = '/home1/junjieyu/mrst-2022b'
    print('matlab engine started')

    train_loader, test_loader =  get_straight_geological_scenario_dataloaders(batch_size=64)
    for rand_batch, rand_labels in train_loader:
        break
    # get model
    path_folder = '../Training_Experiments/model_training/well_trained_model_Jan2023/conditionalVAE/log-gamma10/'
    model_name = 'model_ep25.pt'
    parametrization_model = load(path_folder, model_type = 'conditionalVAE', model_name=model_name)

    # coeffcient for penalty term
    alpha = 0.01
    # define objetive function
    def obj_handle(z):
        obs_loss = two_phase_mismatch(z, eng, MRST_folder, parametrization_model)
        penalty = one_hot_penalty(z, alpha)
        print('obs_loss',obs_loss)
        print('penalty',penalty)

        # for debugging
        with open('obs_loss.txt', 'a') as f:
            # obj_loss
            f.write('mismtach'+'\n')
            f.write(str(obs_loss)+'\n')
            # penalty
            f.write('penalty'+'\n')
            f.write(str(penalty)+'\n')
        return obs_loss + penalty

    def obj_handle_numerical_gradient(z):
        # this function calculate numerical gradient of objective function using central difference
        perturb_size = 1e-3
        # formulate perturbation matrix that used for central difference
        z_dim = z.shape[0]

        # initialize perturbation matxi by repeating z
        perturb_mat = np.repeat(z.reshape(1,-1), z_dim*2, axis=0)
        # perturb_mat = np.zeros((z_dim*2, z_dim))
        for i in range(z_dim):
            perturb_mat[i,i] = z[i] + perturb_size
            perturb_mat[i+z_dim,i] = z[i] - perturb_size

        # calculate mismatch batch
        mismatch_batch = two_phase_mismatch_batch(perturb_mat, eng, MRST_folder, parametrization_model)

        # calculate penalty batch
        penalty_batch = np.zeros((z_dim*2,1))
        for i in range(z_dim*2):
            penalty_batch[i] = one_hot_penalty(perturb_mat[i], alpha)
        obj_batch = mismatch_batch + penalty_batch

        # calculate numerical gradient with central difference
        obj_gradient = np.zeros((z_dim,1))
        for i in range(z_dim):
            obj_gradient[i] = (obj_batch[i] - obj_batch[i+z_dim])/(2*perturb_size)
        obj_gradient = obj_gradient.flatten()
        
        # for debug purpose
        # save a temp.txt file to store obj_gradient, use append mode
        # with open('grad.txt', 'a') as f:
        #     f.write(str(obj_gradient))
        #     f.write('\n')

        # # creata a obj.txt file to store obj value, use write mode
        # with open('obj.txt', 'a') as f:
        #     f.write('penalty_batch\n')
        #     f.write(str(penalty_batch))
        #     f.write('\n')
        #     f.write('mismatch_batch\n')
        #     f.write(str(mismatch_batch))
        #     f.write('\n')
        #     f.write('obj_batch\n')
        #     f.write(str(obj_batch))
        #     f.write('\n')

        # # create a perturb_mat.txt file to store perturb_mat, use write mode
        # with open('perturb_mat.txt', 'a') as f:
        #     f.write(str(perturb_mat))
        #     f.write('\n')
        
        

        # normalize gradient
        # norm_obj_gradient = obj_gradient / np.linalg.norm(obj_gradient)
        return obj_gradient
         
    # define gradient function
    def obj_gradient_handle(z):
        obs_loss_gradient = two_phase_mismatch_gradient(z, eng, MRST_folder, parametrization_model)

        # normalize gradient
        norm_obs_loss_gradient = obs_loss_gradient / np.linalg.norm(obs_loss_gradient)
        print('obs_loss_gradient',norm_obs_loss_gradient)
        penalty_gradient = one_hot_penalty_gradient(z, alpha)
        norm_penalty_gradient = penalty_gradient / np.linalg.norm(penalty_gradient)
        print('penalty_gradient',penalty_gradient)
        norm_grad = (norm_obs_loss_gradient + norm_penalty_gradient).flatten()
        norm_grad = norm_grad / np.linalg.norm(norm_grad)
        # print('grad.shape',grad.shape)
        return norm_grad


    ind = 1
    ref_sample = rand_batch[ind]
    
    # ref case
    two_phase_ref(ref_sample, eng, MRST_folder)

    # get details for ref case
    ref_sample_double = matlab.double(ref_sample.view(1, -1).tolist())
    qWs, qOs, bhp, pressure, sw = eng.two_phase_forward_details(ref_sample_double, MRST_folder, nargout = 5)
    ref_dict = {'qWs':qWs, 'qOs':qOs, 'bhp':bhp, 'pressure':pressure, 'sw':sw}

    initial_img = rand_batch[63-ind].flatten().view(-1,1,64,64)
    initial_label = rand_labels[63-ind].flatten()

    initial_lat_cont_vec =  np.array(parametrization_model.encode(initial_img)['cont'][0].detach().flatten())
    initial_lat_vec = np.concatenate((initial_lat_cont_vec, initial_label))

    z0 = initial_lat_vec

    # get bounds
    cont_size = 10
    disc_size = 4
    cont_lower_bound = np.ones(cont_size) * -3
    cont_upper_bound = np.ones(cont_size) * 3
    disc_lower_bound = np.zeros(disc_size)
    disc_upper_bound = np.ones(disc_size)
    lb = np.concatenate((cont_lower_bound, disc_lower_bound))
    ub = np.concatenate((cont_upper_bound, disc_upper_bound))
    bounds = Bounds(lb, ub, keep_feasible=False)

    A = np.concatenate((np.zeros(cont_size), np.ones(disc_size))).reshape((1, 14))
    cons= LinearConstraint(A, 1, 1, keep_feasible=False)

    options = {'disp':True,'maxiter':30,'finite_diff_rel_step':1e-3}

    # create a history dictionary to store the optimization process
    history = {'x':[],'fun':[],'grad':[]}
    
    def monitor(x, OptimizeResult):
        # get iter num
        iter_num = OptimizeResult.nit
        if iter_num % monitor_interval == 0:

            ## create a figure with three subplots, the left show the ref case, the middle show the initial solution
            ## the right show the current solution
            history['x'].append(x)
            history['fun'].append(OptimizeResult.fun)
            history['grad'].append(OptimizeResult.grad)

            fig1, (ax1, ax2, ax3) = plt.subplots(1, 3)
            fig1.figsize=(30,10)
            # get reference image
            ax1.imshow(ref_sample.detach().numpy().reshape(64,64), cmap='jet', vmin=0, vmax=1)
            ax1.title.set_text('reference image')

            # get initial image
            ax2.imshow(initial_img.detach().numpy().reshape(64,64), cmap='jet', vmin=0, vmax=1)
            ax2.title.set_text('initial image')

            # get current image
            current_perm_latent = x
            current_perm = parametrization_model.decode(torch.tensor(current_perm_latent,dtype = torch.float32).view(1,-1))
            ax3.imshow(current_perm.detach().numpy().reshape(64,64), cmap='jet',vmin=0, vmax=1)
            ax3.set_title('current image')


            # add a title and show current objective value and current discrete label
            # get a rounde value of current discrete label
            round_disc_label = np.round(current_perm_latent[-4:], decimals=2)
            fig1.suptitle('current objective value: '+str(OptimizeResult.fun)+'\n'
                            +'current discrete label: '+str(round_disc_label))

            fig1.tight_layout()
            # save figure
            fig1.savefig(os.path.join(opt_process_folder,'calibrator_monitor'+str(par_iter)+'.png'))
            # close fig1
            plt.close(fig1)

            # create a figure to show the optimization process
            fig2, ax = plt.subplots(1, 1)
            fig2.figsize=(10,10)
            print(history['fun'])
            ax.plot(history['fun'])
            ax.set_xlabel('iteration')
            ax.set_ylabel('objective value')
            ax.set_title('optimization process')
            plt.savefig(os.path.join(opt_process_folder,'optimization_process_'+str(par_iter)+'.png'))
            plt.close(fig2)

            # show history matching plot
            # get details for current case
            current_sample = current_perm.detach().view(1,-1)
            current_sample_double = matlab.double(current_sample.tolist())
            qWs, qOs, bhp, pressure, sw = eng.two_phase_forward_details(current_sample_double, MRST_folder, nargout = 5)
            current_dict = {'qWs':qWs, 'qOs':qOs, 'bhp':bhp, 'pressure':pressure, 'sw':sw}
            # plot history matching plot
            plot_history_matching(ref_dict, current_dict, 
                                  figname = os.path.join(opt_process_folder,
                                                          'history_matching_case'+str(par_iter)+'.png'))

            # plot saturation and pressure
            plot_saturation_pressure(ref_dict, current_dict,
                                        figname = os.path.join(opt_process_folder,
                                                                'saturation_pressure_case'+str(par_iter)+'.png'))


    # without parallelization
    # res = minimize(obj_handle, z0, method='trust-constr', 
    #                jac='3-points', bounds=bounds, constraints=[cons],
    #            options = options, 
    #            callback=monitor)
    # with parallelization
    res = minimize(obj_handle, z0, method='trust-constr', 
                   jac= obj_handle_numerical_gradient,
                   bounds=bounds, constraints=[cons],
               options = options, 
               callback = monitor)
    
    # save history in the opt_process_folder
    with open(opt_process_folder+'/history_case'+str(par_iter)+'.pkl', 'wb') as f:
        pickle.dump(history, f)

    
if __name__ == '__main__':
    for i in range(20):
        tester(i)


    
    
