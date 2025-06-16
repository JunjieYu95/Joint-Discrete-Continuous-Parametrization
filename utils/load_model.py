import json
import torch


from betaVAE.models import betaVAE
from conditionalVAE.models import cVAE


from utils.dataloaders import (get_mnist_dataloaders, get_dsprites_dataloader,
                               get_chairs_dataloader, get_fashion_mnist_dataloaders,get_sis_dataloaders)



def load(path,model_type = 'conditionalVAE',model_name = None, specs_name = None):
    """
    Loads a trained model.

    Parameters
    ----------
    path : string
        Path to folder where model is saved. For example
        './trained_models/mnist/'. Note the path MUST end with a '/'
    """
    if model_name is None:
        path_to_model = path + 'model.pt'
    else:
        path_to_model = path + model_name
    
    if specs_name is None:
        path_to_specs = path + 'specs.json'
    else:
        path_to_specs = path + specs_name


    # Open specs file
    with open(path_to_specs) as specs_file:
        specs = json.load(specs_file)

    # Unpack specs
    dataset = specs["dataset"]
    latent_spec = specs["latent_spec"]

    img_size = (1,64,64)
    
    # Get image size
    if dataset == 'mnist' or dataset == 'fashion_mnist':
        img_size = (1, 32, 32)
    if dataset == 'chairs' or dataset == 'dsprites':
        img_size = (1, 64, 64)
    if dataset == 'celeba':
        img_size = (3, 64, 64)
    if dataset == 'sis':
        img_size = (1,64,64)
    if dataset == 'straight':
        img_size = (1,64,64)
    if dataset == 'straight-rotation-augment':
        img_size = (1,64,64)

    # Get model
    if model_type == 'betaVAE':
        model = betaVAE(img_size=img_size, latent_spec=latent_spec)
    elif model_type == 'conditionalVAE':
        model = cVAE(img_size=img_size, latent_spec=latent_spec)
    model.load_state_dict(torch.load(path_to_model,
                                     map_location=lambda storage, loc: storage))

    return model
