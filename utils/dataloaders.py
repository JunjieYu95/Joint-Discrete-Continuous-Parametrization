import glob
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch
import os
import torch.nn.functional as F
import pickle

def get_channel_dataloaders(batch_size = 32):

    toTensor = transforms.Compose(transforms.ToTensor)
    
    toTensor = None
    channel_dataset_train = ChannelDataset(file_name = 'S-HV2000-s64-shuffle-train.pt',root_dir = '../geo_dataset',transform = toTensor)
    channel_dataset_test = ChannelDataset(file_name = 'S-HV2000-s64-shuffle-test.pt',root_dir = '../geo_dataset',transform = toTensor)

    train_loader = DataLoader(channel_dataset_train, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(channel_dataset_test, batch_size = batch_size, shuffle = True)

    return train_loader, test_loader

def get_fluvial_dataloaders(batch_size = 32):

    toTensor = transforms.Compose(transforms.ToTensor)
    
    toTensor = None
    channel_dataset_train = FluvialDataset(file_name = 'fluvial-train.pt',root_dir = '../geo_dataset',transform = toTensor)
    channel_dataset_test = FluvialDataset(file_name = 'fluvial-test.pt',root_dir = '../geo_dataset',transform = toTensor)

    train_loader = DataLoader(channel_dataset_train, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(channel_dataset_test, batch_size = batch_size, shuffle = True)

    return train_loader, test_loader

def get_gaussian_fluvial_dataloaders(batch_size = 32):

    toTensor = transforms.Compose(transforms.ToTensor)
    
    toTensor = None
    channel_dataset_train = CombinedGaussianFluvialDataset(file_name = 'combined-gaussian-fluvial-train.pt',root_dir = '../geo_dataset',transform = toTensor)
    channel_dataset_test = CombinedGaussianFluvialDataset(file_name = 'combined-gaussian-fluvial-test.pt',root_dir = '../geo_dataset',transform = toTensor)

    train_loader = DataLoader(channel_dataset_train, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(channel_dataset_test, batch_size = batch_size, shuffle = True)

    return train_loader, test_loader

def get_straight_fluvial_dataloaders(batch_size = 32):

    toTensor = transforms.Compose(transforms.ToTensor)
    
    toTensor = None
    channel_dataset_train = CombinedStraightFluvialDataset(file_name = 'straight-fluvial-train.pt',root_dir = '../geo_dataset',transform = toTensor)
    channel_dataset_test = CombinedStraightFluvialDataset(file_name = 'straight-fluvial-test.pt',root_dir = '../geo_dataset',transform = toTensor)

    train_loader = DataLoader(channel_dataset_train, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(channel_dataset_test, batch_size = batch_size, shuffle = True)

    return train_loader, test_loader

def get_ellipse_dataloaders(batch_size = 32):

    toTensor = transforms.Compose(transforms.ToTensor)
    
    toTensor = None
    channel_dataset_train = EllipseDataset(file_name = 'halfellipse-train.pt',root_dir = '../geo_dataset',transform = toTensor)
    channel_dataset_test = EllipseDataset(file_name = 'halfellipse-test.pt',root_dir = '../geo_dataset',transform = toTensor)

    train_loader = DataLoader(channel_dataset_train, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(channel_dataset_test, batch_size = batch_size, shuffle = True)

    return train_loader, test_loader

def get_ellipse_fluvial_dataloaders(batch_size = 32):

    toTensor = transforms.Compose(transforms.ToTensor)
    
    toTensor = None
    channel_dataset_train = CombinedEllipseFluvialDataset(file_name = 'ellipse-fluvial-train.pt',root_dir = '../geo_dataset',transform = toTensor)
    channel_dataset_test = CombinedEllipseFluvialDataset(file_name = 'ellipse-fluvial-test.pt',root_dir = '../geo_dataset',transform = toTensor)

    train_loader = DataLoader(channel_dataset_train, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(channel_dataset_test, batch_size = batch_size, shuffle = True)

    return train_loader, test_loader

def get_sis_dataloaders(batch_size = 32):

    toTensor = transforms.Compose(transforms.ToTensor)
    
    toTensor = None
    channel_dataset_train = SISDataset(file_name = 'uncond-sis-train.pt',root_dir = '../geo_dataset',transform = toTensor)
    channel_dataset_test = SISDataset(file_name = 'uncond-sis-test.pt',root_dir = '../geo_dataset',transform = toTensor)

    train_loader = DataLoader(channel_dataset_train, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(channel_dataset_test, batch_size = batch_size, shuffle = True)

    return train_loader, test_loader

def get_sis_fluvial_dataloaders(batch_size = 32):

    toTensor = transforms.Compose(transforms.ToTensor)
    
    toTensor = None
    channel_dataset_train = SISFluvialDataset(file_name = 'sis-fluvial-train.pt',root_dir = '../geo_dataset',transform = toTensor)
    channel_dataset_test = SISFluvialDataset(file_name = 'sis-fluvial-test.pt',root_dir = '../geo_dataset',transform = toTensor)

    train_loader = DataLoader(channel_dataset_train, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(channel_dataset_test, batch_size = batch_size, shuffle = True)

    return train_loader, test_loader

def get_straight_intersect_dataloaders(batch_size = 128):

    toTensor = transforms.Compose(transforms.ToTensor)
    
    toTensor = None
    channel_dataset_train = StraightIntersectDataset(file_name = 'channel-straight-intersect-shuffle-train.pt',root_dir = '../geo_dataset',transform = toTensor)
    channel_dataset_test = StraightIntersectDataset(file_name = 'channel-straight-intersect-shuffle-test.pt',root_dir = '../geo_dataset',transform = toTensor)

    train_loader = DataLoader(channel_dataset_train, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(channel_dataset_test, batch_size = batch_size, shuffle = True)

    return train_loader, test_loader

def get_straight_dataloaders(batch_size = 128):

    toTensor = transforms.Compose(transforms.ToTensor)
    
    toTensor = None
    channel_dataset_train = StraightIntersectDataset(file_name = 'channel-straight-shuffle-train.pt',root_dir = '../geo_dataset',transform = toTensor)
    channel_dataset_test = StraightIntersectDataset(file_name = 'channel-straight-shuffle-test.pt',root_dir = '../geo_dataset',transform = toTensor)

    train_loader = DataLoader(channel_dataset_train, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(channel_dataset_test, batch_size = batch_size, shuffle = True)

    return train_loader, test_loader

def get_straight_rotation_augment_dataloaders(batch_size = 128):

    toTensor = transforms.Compose(transforms.ToTensor)
    
    toTensor = None
    channel_dataset_train = StraightRotationAugmentDataset(file_name = 'channel-straight-rotation-augment2-train.pt',root_dir = '../geo_dataset',transform = toTensor)
    channel_dataset_test = StraightRotationAugmentDataset(file_name = 'channel-straight-rotation-augment2-test.pt',root_dir = '../geo_dataset',transform = toTensor)

    train_loader = DataLoader(channel_dataset_train, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(channel_dataset_test, batch_size = batch_size, shuffle = True)

    return train_loader, test_loader


def get_straight_generative_augment_dataloaders(batch_size = 128):

    toTensor = transforms.Compose(transforms.ToTensor)
    
    toTensor = None
    channel_dataset_train = StraightGenerativeAugmentDataset(file_name = 'channel-straight-augment-generative-train.pt',root_dir = '../geo_dataset',transform = toTensor)
    channel_dataset_test = StraightGenerativeAugmentDataset(file_name = 'channel-straight-augment-generative-test.pt',root_dir = '../geo_dataset',transform = toTensor)

    train_loader = DataLoader(channel_dataset_train, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(channel_dataset_test, batch_size = batch_size, shuffle = True)

    return train_loader, test_loader

def get_straight_generative_augment_anglelimit_dataloaders(batch_size = 128):

    toTensor = transforms.Compose(transforms.ToTensor)
    
    toTensor = None
    channel_dataset_train = StraightGenerativeAugmentAnglelimitDataset(file_name = 'channel-straight-augment-generative-anglelimit-train.pt',root_dir = '../geo_dataset',transform = toTensor)
    channel_dataset_test = StraightGenerativeAugmentAnglelimitDataset(file_name = 'channel-straight-augment-generative-anglelimit-test.pt',root_dir = '../geo_dataset',transform = toTensor)

    train_loader = DataLoader(channel_dataset_train, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(channel_dataset_test, batch_size = batch_size, shuffle = True)

    return train_loader, test_loader

def get_mandering_generative_augment_anglelimit_dataloaders(batch_size = 128):

    toTensor = transforms.Compose(transforms.ToTensor)
    
    toTensor = None
    channel_dataset_train = ManderingGenerativeAugmentAnglelimitDataset(file_name = 'channel-mandering-augment-generative-anglelimit-train.pt',root_dir = '../geo_dataset',transform = toTensor)
    channel_dataset_test = ManderingGenerativeAugmentAnglelimitDataset(file_name = 'channel-mandering-augment-generative-anglelimit-test.pt',root_dir = '../geo_dataset',transform = toTensor)

    train_loader = DataLoader(channel_dataset_train, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(channel_dataset_test, batch_size = batch_size, shuffle = True)

    return train_loader, test_loader

def get_straight_geological_scenario_dataloaders(batch_size = 128):

    toTensor = transforms.Compose(transforms.ToTensor)
    
    toTensor = None
    channel_dataset_train = StraightGeologicalScenarioDataset(file_name = 'straight-geological-scenario-train.pkl',root_dir = '../geo_dataset',transform = toTensor)
    channel_dataset_test = StraightGeologicalScenarioDataset(file_name = 'straight-geological-scenario-test.pkl',root_dir = '../geo_dataset',transform = toTensor)

    train_loader = DataLoader(channel_dataset_train, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(channel_dataset_test, batch_size = batch_size, shuffle = True)

    return train_loader, test_loader



def get_mnist_dataloaders(batch_size=128, path_to_data='../data'):
    """MNIST dataloader with (32, 32) images."""
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    train_data = datasets.MNIST(path_to_data, train=True, download=True,
                                transform=all_transforms,
                                target_transform=transforms.Compose([
                                 lambda x:torch.tensor([x]), # or just torch.tensor
                                 lambda x:F.one_hot(x,10),
                                 lambda x:torch.squeeze(x),
                                 lambda x:torch.tensor(x,dtype = torch.float)]))

    test_data = datasets.MNIST(path_to_data, train=False,
                               transform=all_transforms,
                               target_transform=transforms.Compose([
                                 lambda x:torch.tensor([x]), # or just torch.tensor
                                 lambda x:F.one_hot(x,10),
                                 lambda x:torch.squeeze(x),
                                 lambda x:torch.tensor(x,dtype = torch.float)]))
                              
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_fashion_mnist_dataloaders(batch_size=128,
                                  path_to_data='../fashion_data'):
    """FashionMNIST dataloader with (32, 32) images."""
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    train_data = datasets.FashionMNIST(path_to_data, train=True, download=True,
                                       transform=all_transforms)
    test_data = datasets.FashionMNIST(path_to_data, train=False,
                                      transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_dsprites_dataloader(batch_size=128,
                            path_to_data='../dsprites-data/dsprites_data.npz'):
    """DSprites dataloader."""
    dsprites_data = DSpritesDataset(path_to_data,
                                    transform=transforms.ToTensor())
    dsprites_loader = DataLoader(dsprites_data, batch_size=batch_size,
                                 shuffle=True)
    return dsprites_loader


def get_chairs_dataloader(batch_size=128,
                          path_to_data='../rendered_chairs_64'):
    """Chairs dataloader. Chairs are center cropped and resized to (64, 64)."""
    all_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    chairs_data = datasets.ImageFolder(root=path_to_data,
                                       transform=all_transforms)
    chairs_loader = DataLoader(chairs_data, batch_size=batch_size,
                               shuffle=True)
    return chairs_loader


def get_chairs_test_dataloader(batch_size=62,
                               path_to_data='../rendered_chairs_64_test'):
    """There are 62 pictures of each chair, so get batches of data containing
    one chair per batch."""
    all_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    chairs_data = datasets.ImageFolder(root=path_to_data,
                                       transform=all_transforms)
    chairs_loader = DataLoader(chairs_data, batch_size=batch_size,
                               shuffle=False)
    return chairs_loader


def get_celeba_dataloader(batch_size=128, path_to_data='../celeba_64'):
    """CelebA dataloader with (64, 64) images."""
    celeba_data = CelebADataset(path_to_data,
                                transform=transforms.ToTensor())
    celeba_loader = DataLoader(celeba_data, batch_size=batch_size,
                               shuffle=True)
    return celeba_loader


class DSpritesDataset(Dataset):
    """D Sprites dataset."""
    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        self.imgs = np.load(path_to_data)['imgs'][::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Each image in the dataset has binary values so multiply by 255 to get
        # pixel values
        sample = self.imgs[idx] * 255
        # Add extra dimension to turn shape into (H, W) -> (H, W, C)
        sample = sample.reshape(sample.shape + (1,))

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0


class CelebADataset(Dataset):
    """CelebA dataset with 64 by 64 images."""
    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        self.img_paths = glob.glob(path_to_data + '/*')[::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        sample = imread(sample_path)

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0

class ChannelDataset(Dataset):
    def __init__(self, file_name, root_dir, transform=None):
        images_path = os.path.join(root_dir, file_name)
        self.channel_frames = torch.load(images_path)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.channel_frames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.channel_frames[idx]
        
        if self.transform:
            sample = self.transform(sample)
        return sample, 0

class FluvialDataset(Dataset):
    def __init__(self, file_name, root_dir, transform=None):
        images_path = os.path.join(root_dir, file_name)
        self.channel_frames = torch.load(images_path)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.channel_frames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.channel_frames[idx]
        
        if self.transform:
            sample = self.transform(sample)
        return sample, 0

class CombinedGaussianFluvialDataset(Dataset):
    def __init__(self, file_name, root_dir, transform=None):
        images_path = os.path.join(root_dir, file_name)
        self.channel_frames = torch.load(images_path)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.channel_frames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.channel_frames[idx]
        
        if self.transform:
            sample = self.transform(sample)
        return sample, 0

class CombinedStraightFluvialDataset(Dataset):
    def __init__(self, file_name, root_dir, transform=None):
        images_path = os.path.join(root_dir, file_name)
        self.channel_frames = torch.load(images_path)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.channel_frames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.channel_frames[idx]
        
        if self.transform:
            sample = self.transform(sample)
        return sample, 0

class EllipseDataset(Dataset):
    def __init__(self, file_name, root_dir, transform=None):
        images_path = os.path.join(root_dir, file_name)
        self.channel_frames = torch.load(images_path)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.channel_frames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.channel_frames[idx]
        
        if self.transform:
            sample = self.transform(sample)
        return sample, 0

class CombinedEllipseFluvialDataset(Dataset):
    def __init__(self, file_name, root_dir, transform=None):
        images_path = os.path.join(root_dir, file_name)
        self.channel_frames = torch.load(images_path)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.channel_frames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.channel_frames[idx]
        
        if self.transform:
            sample = self.transform(sample)
        return sample, 0

class SISDataset(Dataset):
    def __init__(self, file_name, root_dir, transform=None):
        images_path = os.path.join(root_dir, file_name)
        self.channel_frames = torch.load(images_path)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.channel_frames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.channel_frames[idx]
        
        if self.transform:
            sample = self.transform(sample)
        return sample, 0

class SISFluvialDataset(Dataset):
    def __init__(self, file_name, root_dir, transform=None):
        images_path = os.path.join(root_dir, file_name)
        self.channel_frames = torch.load(images_path)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.channel_frames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.channel_frames[idx]
        
        if self.transform:
            sample = self.transform(sample)
        return sample, 0

class StraightIntersectDataset(Dataset):
    def __init__(self, file_name, root_dir, transform=None):
        images_path = os.path.join(root_dir, file_name)
        self.channel_frames = torch.load(images_path)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.channel_frames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.channel_frames[idx]
        
        if self.transform:
            sample = self.transform(sample)
        return sample, 0

class StraightDataset(Dataset):
    def __init__(self, file_name, root_dir, transform=None):
        images_path = os.path.join(root_dir, file_name)
        self.channel_frames = torch.load(images_path)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.channel_frames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.channel_frames[idx]
        
        if self.transform:
            sample = self.transform(sample)
        return sample, 0

class StraightRotationAugmentDataset(Dataset):
    def __init__(self, file_name, root_dir, transform=None):
        images_path = os.path.join(root_dir, file_name)
        self.channel_frames = torch.load(images_path)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.channel_frames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.channel_frames[idx]
        
        if self.transform:
            sample = self.transform(sample)
        return sample, 0

class StraightGenerativeAugmentDataset(Dataset):
    def __init__(self, file_name, root_dir, transform=None):
        images_path = os.path.join(root_dir, file_name)
        self.channel_frames = torch.load(images_path)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.channel_frames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.channel_frames[idx]
        
        if self.transform:
            sample = self.transform(sample)
        return sample, 0

class StraightGenerativeAugmentAnglelimitDataset(Dataset):
    def __init__(self, file_name, root_dir, transform=None):
        images_path = os.path.join(root_dir, file_name)
        self.channel_frames = torch.load(images_path)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.channel_frames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.channel_frames[idx]
        
        if self.transform:
            sample = self.transform(sample)
        return sample, 0

class ManderingGenerativeAugmentAnglelimitDataset(Dataset):
    def __init__(self, file_name, root_dir, transform=None):
        images_path = os.path.join(root_dir, file_name)
        self.channel_frames = torch.load(images_path)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.channel_frames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.channel_frames[idx]
        
        if self.transform:
            sample = self.transform(sample)
        return sample, 0

class StraightGeologicalScenarioDataset(Dataset):

    def __init__(self, file_name, root_dir, transform=None):
        images_path = os.path.join(root_dir, file_name)

        with open(images_path,'rb') as f:
            self.data_pair = pickle.load(f)
            
        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        return len(self.data_pair)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = torch.tensor(self.data_pair[idx][0],dtype=torch.float).view(-1,64,64)
        label = torch.tensor(self.data_pair[idx][1],dtype = torch.float)
        
        if self.transform:
            sample = self.transform(sample)
        return sample, label