import numpy as np
import torch
from .latent_traversals import LatentTraverser
from scipy import stats
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# return  a colormap figure given tensor grid:
def SaveColormapFigure(tensor_grid, cmap = 'jet',name = 'colormapfigure.png',vmin = -0.75,vmax =0.75,colorbar = True):
    plt.figure()
    plt.imshow(tensor_grid[0,:,:].detach().numpy().squeeze(),cmap = cmap,vmin = vmin,vmax = vmax) # use the first channel would be enough, all third channel are the same
    # plt.imshow(img_array,cmap = cmap)
    if colorbar:
        plt.colorbar()
    plt.axis('off')
    plt.savefig(name, dpi = 300, bbox_inches = 'tight')
    plt.close()


class Visualizer():
    def __init__(self, model,use_colormap = True, 
                 include_label = False,
                 seed=None, AutoEncoderTraverse=False):
        """
        Visualizer is used to generate images of samples, reconstructions,
        latent traversals and so on of the trained model.

        Parameters
        ----------
        model : jointvae.models.VAE instance

        Modify: for gamma = 0, VAE becomes AE and traverse latent would no longer constrined in ~(-3sigma,3sigma), for SIS data, is (-40,40)
        use AutoEncoderTraverse to specify this edge case.

        """
        self.model = model
        self.latent_traverser = LatentTraverser(self.model.latent_spec)
        self.save_images = True  # If false, each method returns a tensor
                                 # instead of saving image.
        self.use_colormap = use_colormap
        if seed is not None:
            torch.random.manual_seed(seed)
        self.include_label = include_label

    def reconstructions(self, data, label, size=(8, 8), filename='recon.png'):
        """
        Generates reconstructions of data through the model.

        Parameters
        ----------
        data : torch.Tensor
            Data to be reconstructed. Shape (N, C, H, W)

        size : tuple of ints
            Size of grid on which reconstructions will be plotted. The number
            of rows should be even, so that upper half contains true data and
            bottom half contains reconstructions
        """
        # Plot reconstructions in test mode, i.e. without sampling from latent
        self.model.eval()
        # Pass data through VAE to obtain reconstruction
        with torch.no_grad():
            input_data = Variable(data)
            if self.model.use_cuda:
                input_data = input_data.cuda()
                input_label = label.cuda()
            
            if self.include_label:
                recon_data, _ = self.model(input_data,input_label)
            else:
                recon_data, _ = self.model(input_data)
        self.model.train()

        # Upper half of plot will contain data, bottom half will contain
        # reconstructions
        
        num_images = int(size[0] * size[1] / 2)
        originals = input_data[:num_images].cpu()
        reconstructions = recon_data.view(-1, *self.model.img_size)[:num_images].cpu()
        # If there are fewer examples given than spaces available in grid,
        # augment with blank images
        num_examples = originals.size()[0]
        if num_images > num_examples:
            blank_images = torch.zeros((num_images - num_examples,) + originals.size()[1:])
            originals = torch.cat([originals, blank_images])
            reconstructions = torch.cat([reconstructions, blank_images])

        # Concatenate images and reconstructions
        comparison = torch.cat([originals, reconstructions])
        tensor_grid = make_grid(comparison.data, nrow=size[0],padding = 5, pad_value = 0.5)
        if self.save_images:
            if self.use_colormap:
                SaveColormapFigure(tensor_grid, name=filename,vmin = 0,vmax = 1)
            else:
                save_image(comparison.data, filename, nrow=size[0])
        else:
            return make_grid(comparison.data, nrow=size[0],padding = 5, pad_value = 0.5)

    def samples(self, size=(8, 8), filename='samples.png'):
        """
        Generates samples from learned distribution by sampling prior and
        decoding.

        size : tuple of ints
        """
        # Get prior samples from latent distribution
        cached_sample_prior = self.latent_traverser.sample_prior
        self.latent_traverser.sample_prior = True
        prior_samples = self.latent_traverser.traverse_grid(size=size)
        self.latent_traverser.sample_prior = cached_sample_prior

        # Map samples through decoder
        generated = self._decode_latents(prior_samples)
        tensor_grid = make_grid(generated.data, nrow=size[1],padding = 15, pad_value = 0.5)

        if self.save_images:
            if self.use_colormap:
                SaveColormapFigure(tensor_grid, name=filename,vmin = 0,vmax = 1)
            else:
                save_image(generated.data, filename, nrow=size[1])
        else:
            return make_grid(generated.data, nrow=size[1])
    
    def samples_with_fixed_scenarios(self, size=(8, 8), scenario_index = 0, filename='samples.png'):
        # Get prior samples from latent distribution
        cached_sample_prior = self.latent_traverser.sample_prior
        self.latent_traverser.sample_prior = True
        prior_samples = self.latent_traverser.traverse_grid(size=size)
        self.latent_traverser.sample_prior = cached_sample_prior

        # manually fix the discrete latent
        prior_samples[:,self.model.latent_cont_dim:] = 0
        prior_samples[:,self.model.latent_cont_dim+scenario_index] = 1

        # Map samples through decoder
        generated = self._decode_latents(prior_samples)
        tensor_grid = make_grid(generated.data, nrow=size[1],padding = 5, pad_value = 0.5)

        if self.save_images:
            if self.use_colormap:
                SaveColormapFigure(tensor_grid, name=filename,vmin = 0,vmax = 1)
            else:
                save_image(generated.data, filename, nrow=size[1])
        else:
            return make_grid(generated.data, nrow=size[1])


    def latent_traversal_line(self, cont_idx=None, disc_idx=None, size=8,
                              filename='traversal_line.png'):
        """
        Generates an image traversal through a latent dimension.

        Parameters
        ----------
        See viz.latent_traversals.LatentTraverser.traverse_line for parameter
        documentation.
        """
        # Generate latent traversal
        latent_samples = self.latent_traverser.traverse_line(cont_idx=cont_idx,
                                                             disc_idx=disc_idx,
                                                             size=size)

        # Map samples through decoder
        generated = self._decode_latents(latent_samples)
        tensor_grid =  make_grid(generated.data, nrow=size)

        if self.save_images:
            if self.use_colormap:
                SaveColormapFigure(tensor_grid, name=filename,vmin = 0,vmax = 1)
            else:
                save_image(generated.data, filename, nrow=size)
        else:
            return make_grid(generated.data, nrow=size)

    def latent_traversal_grid(self, cont_idx=None, cont_axis=None,
                              disc_idx=None, disc_axis=None, size=(5, 5),
                              filename='traversal_grid.png'):
        """
        Generates a grid of image traversals through two latent dimensions.

        Parameters
        ----------
        See viz.latent_traversals.LatentTraverser.traverse_grid for parameter
        documentation.
        """
        # Generate latent traversal
        latent_samples = self.latent_traverser.traverse_grid(cont_idx=cont_idx,
                                                             cont_axis=cont_axis,
                                                             disc_idx=disc_idx,
                                                             disc_axis=disc_axis,
                                                             size=size)

        # Map samples through decoder
        generated = self._decode_latents(latent_samples)
        tensor_grid = make_grid(generated.data, nrow=size[1])
        if self.save_images:
            if self.use_colormap:
                SaveColormapFigure(tensor_grid,name=filename,vmin = 0,vmax = 1)
            else:
                save_image(generated.data, filename, nrow=size[1])
        else:
            return make_grid(generated.data, nrow=size[1])

    def latent_traversal_grid_diff2(self, cont_idx=None, cont_axis=None,
                              disc_idx=None, disc_axis=None, size=(5, 5),
                              filename='traversal_grid.png'):
        """
        Generates a grid of image traversals through two latent dimensions.

        Parameters
        ----------
        See viz.latent_traversals.LatentTraverser.traverse_grid for parameter
        documentation.
        """
        # Generate latent traversal
        latent_samples = self.latent_traverser.traverse_grid(cont_idx=cont_idx,
                                                             cont_axis=cont_axis,
                                                             disc_idx=disc_idx,
                                                             disc_axis=disc_axis,
                                                             size=size)

        # Map samples through decoder
        generated = self._decode_latents(latent_samples)
        generated_diff = generated.data.clone()
        num_samples = size[0]*size[1]
        for i in range(0,size[0]):
            for j in range(0,size[1]):
                if j == 0:
                    generated_diff[i*size[1],0,:,:] = generated[i*size[1],0,:,:]*0
                else:
                    generated_diff[i*size[1]+j,0,:,:] = generated[i*size[1]+j,0,:,:]-generated[i*size[1]+j-1,0,:,:]

        if self.save_images:
            save_image(generated_diff, filename, nrow=size[1])
        else:
            return make_grid(generated_diff, nrow=size[1])

    def latent_traversal_grid_diff(self, cont_idx=None, cont_axis=None,
                              disc_idx=None, disc_axis=None, size=(5, 5),
                              filename='traversal_grid_diff.png'):
        """
        Generates a grid of image traversals through two latent dimensions.

        Parameters
        ----------
        See viz.latent_traversals.LatentTraverser.traverse_grid for parameter
        documentation.
        """
        # Generate latent traversal
        latent_samples = self.latent_traverser.traverse_grid(cont_idx=cont_idx,
                                                             cont_axis=cont_axis,
                                                             disc_idx=disc_idx,
                                                             disc_axis=disc_axis,
                                                             size=size)

        # Map samples through decoder
        generated = self._decode_latents(latent_samples)
        generated_diff = generated.data.clone()
        for i in range(0,size[0]):
            for j in range(0,size[1]):
                if j == 0:
                    generated_diff[i*size[1],0,:,:] = generated[i*size[1],0,:,:]*0
                else:
                    generated_diff[i*size[1]+j,0,:,:] = generated[i*size[1]+j,0,:,:]-generated[i*size[1],0,:,:]

        tensor_grid = make_grid(generated_diff, nrow=size[1], padding = 5, pad_value = -0.6)
        if self.save_images:
            if self.use_colormap:
                SaveColormapFigure(tensor_grid, cmap='coolwarm',name=filename)
            else:
                save_image(generated_diff, filename, nrow=size[1])
        else:
            return make_grid(generated_diff, nrow=size[1])



    def all_latent_traversals(self, size=8, filename='all_traversals.png'):
        """
        Traverses all latent dimensions one by one and plots a grid of images
        where each row corresponds to a latent traversal of one latent
        dimension.

        Parameters
        ----------
        size : int
            Number of samples for each latent traversal.
        """
        latent_samples = []

        # Perform line traversal of every continuous and discrete latent
        for cont_idx in range(self.model.latent_cont_dim):
            latent_samples.append(self.latent_traverser.traverse_line(cont_idx=cont_idx,
                                                                      disc_idx=None,
                                                                      size=size))

        for disc_idx in range(self.model.num_disc_latents):
            latent_samples.append(self.latent_traverser.traverse_line(cont_idx=None,
                                                                      disc_idx=disc_idx,
                                                                      size=size))

        # Decode samples
        generated = self._decode_latents(torch.cat(latent_samples, dim=0))
        tensor_grid = make_grid(generated.data, nrow=size)

        if self.save_images:
            if self.use_colormap:
                SaveColormapFigure(tensor_grid, name = filename,vmin = 0,vmax = 1)
            else:
                save_image(generated.data, filename, nrow=size)
        else:
            return make_grid(generated.data, nrow=size,padding=5, pad_value = 0.5)

    def all_latent_traversals_diff(self, size=8, filename='all_traversals.png'):
        """
        Traverses all latent dimensions one by one and plots a grid of images
        where each row corresponds to a latent traversal of one latent
        dimension.

        Parameters
        ----------
        size : int
            Number of samples for each latent traversal.
        """
        latent_samples = []

        # Perform line traversal of every continuous and discrete latent
        for cont_idx in range(self.model.latent_cont_dim):
            latent_samples.append(self.latent_traverser.traverse_line(cont_idx=cont_idx,
                                                                      disc_idx=None,
                                                                      size=size))

        for disc_idx in range(self.model.num_disc_latents):
            latent_samples.append(self.latent_traverser.traverse_line(cont_idx=None,
                                                                      disc_idx=disc_idx,
                                                                      size=size))

        # Decode samples
        generated = self._decode_latents(torch.cat(latent_samples, dim=0))
        generated_diff = generated.data.clone()
        latent_dim = self.model.latent_cont_dim+self.model.num_disc_latents
        ref_loc = size // 2
        for i in range(0,latent_dim):
            for j in range(0,size):
                if j == 0:
                    # generated_diff[i*size+j,0,:,:] = generated[i*size+j,0,:,:]*0
                    generated_diff[i*size,0,:,:] = generated[i*size,0,:,:]*0
                else:
                    generated_diff[i*size+j,0,:,:] = generated[i*size+j,0,:,:]-generated[i*size,0,:,:]
                    # generated_diff[i*size+j,0,:,:] = generated[i*size+j,0,:,:]-generated[i*size+ref_loc,0,:,:]


        tensor_grid = make_grid(generated_diff, nrow=size, padding = 5, pad_value = -0.6)
        if self.save_images:
            if self.use_colormap:
                SaveColormapFigure(tensor_grid, name = filename,vmin = -0.75,vmax = 0.75)
            else:
                save_image(generated_diff, filename, nrow=size)
        else:
            return make_grid(generated_diff, nrow=size, padding = 5, pad_value = -0.6)

    def _decode_latents(self, latent_samples):
        """
        Decodes latent samples into images.

        Parameters
        ----------
        latent_samples : torch.autograd.Variable
            Samples from latent distribution. Shape (N, L) where L is dimension
            of latent distribution.
        """
        latent_samples = Variable(latent_samples)
        if self.model.use_cuda:
            latent_samples = latent_samples.cuda()
        return self.model.decode(latent_samples).cpu()


def reorder_img(orig_img, reorder, by_row=True, img_size=(3, 32, 32), padding=2):
    """
    Reorders rows or columns of an image grid.

    Parameters
    ----------
    orig_img : torch.Tensor
        Original image. Shape (channels, width, height)

    reorder : list of ints
        List corresponding to desired permutation of rows or columns

    by_row : bool
        If True reorders rows, otherwise reorders columns

    img_size : tuple of ints
        Image size following pytorch convention

    padding : int
        Number of pixels used to pad in torchvision.utils.make_grid
    """
    reordered_img = torch.zeros(orig_img.size())
    _, height, width = img_size

    for new_idx, old_idx in enumerate(reorder):
        if by_row:
            start_pix_new = new_idx * (padding + height) + padding
            start_pix_old = old_idx * (padding + height) + padding
            reordered_img[:, start_pix_new:start_pix_new + height, :] = orig_img[:, start_pix_old:start_pix_old + height, :]
        else:
            start_pix_new = new_idx * (padding + width) + padding
            start_pix_old = old_idx * (padding + width) + padding
            reordered_img[:, :, start_pix_new:start_pix_new + width] = orig_img[:, :, start_pix_old:start_pix_old + width]

    return reordered_img