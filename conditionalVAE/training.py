import imageio
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.utils import make_grid
from utils.dataloaders import (get_channel_dataloaders,get_fluvial_dataloaders,get_gaussian_fluvial_dataloaders,
get_straight_fluvial_dataloaders,get_ellipse_dataloaders,get_ellipse_fluvial_dataloaders,get_sis_dataloaders,get_sis_fluvial_dataloaders)
import os
import shutil
import matplotlib.pyplot as plt
import json

EPS = 1e-12


class Trainer():
    def __init__(self, model, optimizer, cont_capacity=None, dataset = 'sis',
                 disc_capacity=None, print_loss_every=10, record_loss_every=5,
                 use_cuda=False, log_folder = 'log'):
        """
        Class to handle training of model.

        Parameters
        ----------
        model : jointvae.models.VAE instance

        optimizer : torch.optim.Optimizer instance

        cont_capacity : tuple (float, float, int, float) or None
            Tuple containing (min_capacity, max_capacity, num_iters, gamma_z).
            Parameters to control the capacity of the continuous latent
            channels. Cannot be None if model.is_continuous is True.

        disc_capacity : tuple (float, float, int, float) or None
            Tuple containing (min_capacity, max_capacity, num_iters, gamma_c).
            Parameters to control the capacity of the discrete latent channels.
            Cannot be None if model.is_discrete is True.

        print_loss_every : int
            Frequency with which loss is printed during training.

        record_loss_every : int
            Frequency with which loss is recorded during training.

        use_cuda : bool
            If True moves model and training to GPU.
        """
        self.model = model
        self.optimizer = optimizer
        self.cont_capacity = cont_capacity
        self.disc_capacity = disc_capacity
        self.print_loss_every = print_loss_every
        self.record_loss_every = record_loss_every
        self.use_cuda = use_cuda
        self.log_folder = log_folder
        self.dataset = dataset

        if self.model.is_continuous and self.cont_capacity is None:
            raise RuntimeError("Model is continuous but cont_capacity not provided.")

        if self.model.is_discrete and self.disc_capacity is None:
            raise RuntimeError("Model is discrete but disc_capacity not provided.")

        if self.use_cuda:
            self.model.cuda()

        # Initialize attributes
        self.num_steps = 0
        self.batch_size = None

        # this losses refer to train losses
        self.losses = {'loss': [],
                       'recon_loss': [],
                       'kl_loss': []}

        # add test lossses, original repository does not store test losses               
        self.losses_test = {'loss': [],
                       'recon_loss': [],
                       'kl_loss': []}

        # Keep track of divergence values for each latent variable
        if self.model.is_continuous:
            self.losses['kl_loss_cont'] = []
            # For every dimension of continuous latent variables
            for i in range(self.model.latent_spec['cont']):
                self.losses['kl_loss_cont_' + str(i)] = []

        if self.model.is_discrete:
            self.losses['kl_loss_disc'] = []
            # For every discrete latent variable
            for i in range(len(self.model.latent_spec['disc'])):
                self.losses['kl_loss_disc_' + str(i)] = []

    def train(self, train_loader, test_loader, epochs=10, save_training_gif=None, monitor_interval = 20):
        """
        Trains the model.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader

        epochs : int
            Number of epochs to train the model for.

        save_training_gif : None or tuple (string, Visualizer instance)
            If not None, will use visualizer object to create image of samples
            after every epoch and will save gif of these at location specified
            by string. Note that string should end with '.gif'.
        """
        if save_training_gif is not None:
            training_progress_images = []

        self.batch_size = train_loader.batch_size
        self.model.train()
        if os.path.exists(self.log_folder):
            shutil.rmtree(self.log_folder)
        os.mkdir(self.log_folder)

        specs = {}
        specs['latent_spec'] = self.model.latent_spec
        specs['dataset'] = self.dataset
        specs["cont_capacity"] = self.cont_capacity
        specs["disc_capacity"] = self.disc_capacity
        with open(str(self.log_folder)+'/specs.json','w') as outfile:
            json.dump(specs,outfile)
        
        for epoch in range(epochs):
            mean_epoch_loss = self._train_epoch(train_loader, test_loader)
            print('Epoch: {} Average Training loss: {:.2f}'.format(epoch + 1,
                                                          self.batch_size * self.model.num_pixels * mean_epoch_loss))

            if save_training_gif is not None:
                # Generate batch of images and convert to grid
                viz = save_training_gif[1]
                # viz.save_images = False
                # img_grid = viz.all_latent_traversals(size=10)
                # # Convert to numpy and transpose axes to fit imageio convention
                # # i.e. (width, height, channels)
                # img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
                # img_grid = (img_grid*255.0).astype('uint8') # without this, i HAVE WARNING: Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
                # # Add image grid to training progress
                # training_progress_images.append(img_grid)

            if save_training_gif is not None and save_training_gif[0] is not None:
                imageio.mimsave(save_training_gif[0], training_progress_images,
                                fps=5)

            ## viualization during training
            # Get a batch of data
            if epoch%monitor_interval== 0:
                torch.save(self.model.state_dict(),str(self.log_folder)+'/model_ep'+str(epoch)+'.pt')
                # # save reconstruction and traverse image during training for monitoring process
                for test_batch, test_labels in test_loader:
                    break
                if self.use_cuda:
                    test_batch = test_batch.cuda()
                    test_labels = test_labels.cuda()

                viz.save_images = True

                recon = viz.reconstructions(test_batch,test_labels, filename=str(self.log_folder)+'/reconstruction-ep'+str(epoch)+'.png')

                # traversals = viz.all_latent_traversals(size=10,filename=str(self.log_folder)+'/all_traverse-ep'+str(epoch)+'.png')

                # traverse over continuous variables
                # for ci in range(self.model.latent_cont_dim):
                #     viz.latent_traversal_grid(cont_idx=ci, cont_axis=1, disc_idx=None, disc_axis=None, size=(10, 10),filename=str(self.log_folder)+'/latent_traverse_cont'+str(ci)+'-ep'+str(epoch)+'.png')
                #     viz.latent_traversal_grid_diff(cont_idx=ci, cont_axis=1, disc_idx=None, disc_axis=None, size=(10, 10),filename=str(self.log_folder)+'/latent_traverse_diff_cont'+str(ci)+'-ep'+str(epoch)+'.png')
                
                # for di in range(self.model.latent_disc_dim):
                #     # traverse over discrete variables
                #     viz.samples_with_fixed_scenarios(scenario_index = di, filename=str(self.log_folder)+'/latent_traverse_disc'+str(di)+'-ep'+str(epoch)+'.png')

                # show current loss curve
                with torch.no_grad():
                    plt.figure()
                    plt.plot(self.losses['loss'],'r-o')
                    plt.plot(self.losses['kl_loss'],'r-^')
                    plt.plot(self.losses['recon_loss'],'r-s')
                    plt.plot(self.losses_test['loss'],'b--o')
                    plt.plot(self.losses_test['kl_loss'],'b--^')
                    plt.plot(self.losses_test['recon_loss'],'b--s')
                    plt.legend(['total_train_loss','kl_train_loss','reconstruction_train_loss','total_test_loss','kl_test_loss','reconstruction_test_loss'])
                    plt.xlabel('Iterations')
                    plt.ylabel('loss')
                    plt.savefig(str(self.log_folder)+'/loss.png', dpi = 300, box_inches = 'tight')
                    plt.close()

                    plt.figure()
                    plt.plot(np.array(self.losses['loss'])/self.model.num_pixels,'r-')
                    plt.plot(np.array(self.losses_test['loss'])/self.model.num_pixels,'b--')
                    plt.legend(['train_loss','test_loss'])
                    plt.xlabel('Iterations')
                    plt.ylabel('loss')
                    plt.savefig(str(self.log_folder)+'/loss-simple.png', dpi = 300, box_inches = 'tight')
                    plt.close()
                    # for di in range(2):
                #     viz.latent_traversal_grid(cont_idx=None, cont_axis=None, disc_idx=di, disc_axis=0, size=(10, 10),filename='temp/latent_traverse_disc'+str(di)+'-ep'+str(epoch)+'.png')

                # viz.latent_traversal_grid(cont_idx=0, cont_axis=1, disc_idx=1, disc_axis=0, size=(10, 10),filename='temp/latent_traverse_cont0-lat1-ep'+str(epoch)+'.png')

                # viz.latent_traversal_grid(cont_idx=1, cont_axis=1, disc_idx=0, disc_axis=0, size=(10, 10),filename='temp/latent_traverse_cont1-lat0-ep'+str(epoch)+'.png')

                # viz.latent_traversal_grid(cont_idx=None, cont_axis=None, disc_idx=0, disc_axis=1, size=(10, 10),filename='temp/latent_traverse_lat1-ep'+str(epoch)+'.png')
                # print(recon.shape)


    def _train_epoch(self, train_loader, test_loader):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
        """
        epoch_loss = 0.
        print_every_loss = 0.  # Keeps track of loss to print every
                               # self.print_loss_every
        self.model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            self.model.train()
            iter_loss = self._train_iteration(data, labels)
            epoch_loss += iter_loss
            print_every_loss += iter_loss
            # Print loss info every self.print_loss_every iteration
            if batch_idx % self.print_loss_every == 0:
                if batch_idx == 0:
                    mean_loss = print_every_loss
                else:
                    mean_loss = print_every_loss / self.print_loss_every
                print('{}/{}\tLoss: {:.3f}'.format(batch_idx * len(data),
                                                  len(train_loader.dataset),
                                                  self.model.num_pixels * mean_loss))
                print_every_loss = 0.
            
            # Also updade the test loss during each iteration
            for test_batch, test_labels in test_loader:
                break
            if self.use_cuda:
                test_batch = test_batch.cuda()
                test_labels = test_labels.cuda()
            # set to evaluate model first
            self.model.eval()
            with torch.no_grad():
                test_recon, latent_dist = self.model(test_batch,test_labels)
                test_losses = self._loss_function(test_batch, test_recon, latent_dist)


            # Return mean epoch loss
        return epoch_loss / len(train_loader.dataset)

    def _train_iteration(self, data,labels):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            A batch of data. Shape (N, C, H, W)
        """
        self.num_steps += 1
        if self.use_cuda:
            data = data.cuda()
            labels = labels.cuda()

        self.model.train()
        self.optimizer.zero_grad()
        recon_batch, latent_dist = self.model(data, labels)
        loss = self._loss_function(data, recon_batch, latent_dist)
        loss.backward()
        self.optimizer.step()

        train_loss = loss.item()

        return train_loss

    def _loss_function(self, data, recon_data, latent_dist):
        """
        Calculates loss for a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            Input data (e.g. batch of images). Should have shape (N, C, H, W)

        recon_data : torch.Tensor
            Reconstructed data. Should have shape (N, C, H, W)

        latent_dist : dict
            Dict with keys 'cont' or 'disc' or both containing the parameters
            of the latent distributions as values.
        """
        # Reconstruction loss is pixel wise cross-entropy
        recon_loss = F.binary_cross_entropy(recon_data.view(-1, self.model.num_pixels),
                                            data.view(-1, self.model.num_pixels))
        # F.binary_cross_entropy takes mean over pixels, so unnormalise this
        recon_loss *= self.model.num_pixels

        # Calculate KL divergences
        kl_cont_loss = 0  # Used to compute capacity loss (but not a loss in itself)
        kl_disc_loss = 0  # Used to compute capacity loss (but not a loss in itself)
        cont_capacity_loss = 0
        disc_capacity_loss = 0

        if self.model.is_continuous:
            # Calculate KL divergence
            mean, logvar = latent_dist['cont']
            kl_cont_loss = self._kl_normal_loss(mean, logvar)
            # Linearly increase capacity of continuous channels
            cont_min, cont_max, cont_num_iters, cont_gamma = \
                self.cont_capacity
            # Increase continuous capacity without exceeding cont_max
            cont_cap_current = (cont_max - cont_min) * self.num_steps / float(cont_num_iters) + cont_min
            cont_cap_current = min(cont_cap_current, cont_max)
            # Calculate continuous capacity loss
            cont_capacity_loss = cont_gamma * torch.abs(cont_cap_current - kl_cont_loss)

        if self.model.is_discrete:
            # Calculate KL divergence
            kl_disc_loss = self._kl_multiple_discrete_loss(latent_dist['disc'])
            # Linearly increase capacity of discrete channels
            disc_min, disc_max, disc_num_iters, disc_gamma = \
                self.disc_capacity
            # Increase discrete capacity without exceeding disc_max or theoretical
            # maximum (i.e. sum of log of dimension of each discrete variable)
            disc_cap_current = (disc_max - disc_min) * self.num_steps / float(disc_num_iters) + disc_min
            disc_cap_current = min(disc_cap_current, disc_max)
            # Require float conversion here to not end up with numpy float
            disc_theoretical_max = sum([float(np.log(disc_dim)) for disc_dim in self.model.latent_spec['disc']])
            disc_cap_current = min(disc_cap_current, disc_theoretical_max)
            # Calculate discrete capacity loss
            disc_capacity_loss = disc_gamma * torch.abs(disc_cap_current - kl_disc_loss)

        # Calculate total kl value to record it
        kl_loss = kl_cont_loss + kl_disc_loss
        # print()
        # print(kl_loss)
        # print(recon_loss)
        # print(kl_loss)
        # print(cont_capacity_loss)
        # Calculate total loss
        total_loss = recon_loss + cont_capacity_loss + disc_capacity_loss

        # Record losses
        if self.model.training and self.num_steps % self.record_loss_every == 1:
            # print('train',self.num_steps)
            self.losses['recon_loss'].append(recon_loss.item())
            self.losses['kl_loss'].append(cont_capacity_loss + disc_capacity_loss)
            self.losses['loss'].append(total_loss.item())

        if (not self.model.training) and self.num_steps % self.record_loss_every == 1:
            # print('test',self.num_steps)
            self.losses_test['recon_loss'].append(recon_loss.item())
            self.losses_test['kl_loss'].append(cont_capacity_loss + disc_capacity_loss)
            self.losses_test['loss'].append(total_loss.item())

        # To avoid large losses normalise by number of pixels
        return total_loss / self.model.num_pixels

    def _kl_normal_loss(self, mean, logvar):
        """
        Calculates the KL divergence between a normal distribution with
        diagonal covariance and a unit normal distribution.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (N, D) where D is dimension
            of distribution.

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (N, D)
        """
        # Calculate KL divergence
        kl_values = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        # Mean KL divergence across batch for each latent variable
        kl_means = torch.mean(kl_values, dim=0)
        # KL loss is sum of mean KL of each latent variable
        kl_loss = torch.sum(kl_means)

        # Record losses
        if self.model.training and self.num_steps % self.record_loss_every == 1:
            self.losses['kl_loss_cont'].append(kl_loss.item())
            for i in range(self.model.latent_spec['cont']):
                self.losses['kl_loss_cont_' + str(i)].append(kl_means[i].item())

        return kl_loss

    def _kl_multiple_discrete_loss(self, alphas):
        """
        Calculates the KL divergence between a set of categorical distributions
        and a set of uniform categorical distributions.

        Parameters
        ----------
        alphas : list
            List of the alpha parameters of a categorical (or gumbel-softmax)
            distribution. For example, if the categorical atent distribution of
            the model has dimensions [2, 5, 10] then alphas will contain 3
            torch.Tensor instances with the parameters for each of
            the distributions. Each of these will have shape (N, D).
        """
        # Calculate kl losses for each discrete latent
        kl_losses = [self._kl_discrete_loss(alpha) for alpha in alphas]

        # Total loss is sum of kl loss for each discrete latent
        kl_loss = torch.sum(torch.cat(kl_losses))

        # Record losses
        if self.model.training and self.num_steps % self.record_loss_every == 1:
            self.losses['kl_loss_disc'].append(kl_loss.item())
            for i in range(len(alphas)):
                self.losses['kl_loss_disc_' + str(i)].append(kl_losses[i].item())

        return kl_loss

    def _kl_discrete_loss(self, alpha):
        """
        Calculates the KL divergence between a categorical distribution and a
        uniform categorical distribution.

        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the categorical or gumbel-softmax distribution.
            Shape (N, D)
        """
        disc_dim = int(alpha.size()[-1])
        log_dim = torch.Tensor([np.log(disc_dim)])
        if self.use_cuda:
            log_dim = log_dim.cuda()
        # Calculate negative entropy of each row
        neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
        # Take mean of negative entropy across batch
        mean_neg_entropy = torch.mean(neg_entropy, dim=0)
        # KL loss of alpha with uniform categorical variable
        kl_loss = log_dim + mean_neg_entropy
        return kl_loss
