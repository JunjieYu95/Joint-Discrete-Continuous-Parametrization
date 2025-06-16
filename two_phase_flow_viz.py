import matplotlib.pyplot as plt
import numpy as np

def plot_history_matching(ref_dict, sim_dict, figname='history_match.png'):
    '''
    ref_dict:  {
                qWs: (numWell, timesteps)
                qOs: (numWell, timesteps)
                bhp: (numWell, timesteps)
                }
    '''
    num_wells = len(ref_dict['qWs'])
    # assuming only one injection well here 
    num_inj = 1
    num_prod = num_wells - num_inj

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(15, 20))
    
    # bhp for injection well
    ax1.plot(ref_dict['bhp'][0], 'o', label='obs_inj'+'_'+str(0), color = 'b')
    ax1.plot(sim_dict['bhp'][0],  '-', label='sim_inj'+'_'+str(0), color = 'b')
    ax1.set_title('BHP for injection well')
    ax1.set_ylabel('BHP (bar)')
    ax1.set_xlabel('Time steps')

    # qWs for production wells
    # creata a list of colors from default color cycle
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = list(color_cycle)

    for p in range(1,num_prod):
        ax2.plot(-np.array(ref_dict['qWs'][p]), 'o',label='obs_prod_'+str(p), color = colors[p])
        ax2.plot(-np.array(sim_dict['qWs'][p]),  '-', label='sim_prod_'+str(p), color = colors[p])
    ax2.set_title('Oil rate for production wells')
    ax2.set_ylabel('Oil rate (m3/day)')
    ax2.set_xlabel('Time steps')

    # qOs for production wells
    for p in range(1,num_prod):
        ax3.plot(-np.array(ref_dict['qOs'][p]), 'o', label='obs_prod_'+str(p), color = colors[p])
        ax3.plot(-np.array(sim_dict['qOs'][p]),  '-', label='sim_prod_'+str(p), color = colors[p])
    ax3.set_title('Water rate for production wells')
    ax3.set_ylabel('Water rate (m3/day)')
    ax3.set_xlabel('Time steps')

    plt.legend()
    # increase font size
    # save figure
    fig.savefig(figname, dpi=300, bbox_inches='tight')
    # close figure
    plt.close(fig)

def plot_saturation_pressure(ref_dict, sim_dict, figname='final_saturation_pressure.png'):
    '''
    ref_dict:  {
                Sw: (grid_size, timesteps)
                So: (grid_size, timesteps)
                }
    '''
    grid_size = np.sqrt(np.array(ref_dict['sw']).shape[0]).astype(int)
    # create subplots wih 2*3 grid
    fig, axes = plt.subplots(2,3, figsize=(15, 10))
    
    # pressure

    p_ref = np.array(ref_dict['pressure'])[:,-1]
    p_sim = np.array(sim_dict['pressure'])[:,-1]

    p_diff = p_ref - p_sim
    # reshape to 2D
    p_ref = np.reshape(p_ref, (grid_size, grid_size))
    p_sim = np.reshape(p_sim, (grid_size, grid_size))
    p_diff = np.reshape(p_diff, (grid_size, grid_size))

    # get max and min values for pressure
    p_max = max(np.max(p_ref), np.max(p_sim))
    p_min = min(np.min(p_ref), np.min(p_sim))

    # plot pressure
    axes[0,0].imshow(p_ref, cmap='jet', vmin=p_min, vmax=p_max)
    axes[0,0].set_title('Reference pressure')
    axes[0,1].imshow(p_sim, cmap='jet', vmin=p_min, vmax=p_max)
    axes[0,1].set_title('Simulated pressure')
    axes[0,2].imshow(p_diff, cmap='jet', vmin=p_min, vmax=p_max)
    axes[0,2].set_title('Difference pressure')

    # saturation
    Sw_ref = np.array(ref_dict['sw'])[:,-1]
    Sw_sim = np.array(sim_dict['sw'])[:,-1]
    # transfer to numpy array
    Sw_ref = np.array(Sw_ref)
    Sw_sim = np.array(Sw_sim)
    Sw_diff = Sw_ref - Sw_sim
    # reshape to 2D
    Sw_ref = np.reshape(Sw_ref, (grid_size, grid_size))
    Sw_sim = np.reshape(Sw_sim, (grid_size, grid_size))
    Sw_diff = np.reshape(Sw_diff, (grid_size, grid_size))

    axes[1,0].imshow(Sw_ref, cmap='jet', vmin=0, vmax=1)
    axes[1,0].set_title('Reference sw')
    axes[1,1].imshow(Sw_sim, cmap='jet', vmin=0, vmax=1)
    axes[1,1].set_title('Simulated Sw')
    axes[1,2].imshow(Sw_diff, cmap='jet', vmin=0, vmax=1)
    axes[1,2].set_title('Difference sw')

    # remove ticks
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    # save figure
    fig.savefig(figname, dpi=300, bbox_inches='tight')
    # close figure
    plt.close(fig)
    




