
from typing import Dict, Any, Tuple

from pathlib import Path

import torch
import torch.utils.data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.transforms.functional import crop

import matplotlib.pyplot as plt
import numpy as np

# This is the path to the folder containing the images
#DEFAULT_IMAGE_FOLDER_PATH = Path('dataset/singleaperture/')
DEFAULT_IMAGE_FOLDER_PATH = Path('dataset/singleaperture_noaug_classes/')

def crop512(image):
    return crop(image, 0, 0, 512, 512)

def get_transforms(grayscale: bool = False):
    """
    This function returns the transforms that are applied to the images when they are loaded.
    """
    # Set up the transforms on load of the data
    if grayscale:
        train_transforms = transforms.Compose(
            [
                #transforms.CenterCrop(crop_size),
                #transforms.Resize(resize_size),
                transforms.Grayscale(),
                transforms.Lambda(crop512),
                transforms.ToTensor()
            ]
        )
        # In this case, as we aren't doing any kind of random augmentation we 
        # can use the same transforms for the test data as the train data
        test_transforms = train_transforms
        valid_transforms = train_transforms
    else:
        train_transforms = transforms.Compose(
            [
                #transforms.CenterCrop(crop_size),
                #transforms.Resize(resize_size),
                transforms.Grayscale(),
                transforms.Lambda(crop512),
                transforms.ToTensor()
            ]
        )
        # In this case, as we aren't doing any kind of random augmentation we 
        # can use the same transforms for the test data as the train data
        test_transforms = train_transforms
        valid_transforms = train_transforms
    return train_transforms, test_transforms, valid_transforms


def load_image_targets_from_csv(csv_path: Path, header: bool = True) -> Dict[str, Any]:
    """
    This function loads the image targets from a csv file. It assumes that the csv file
    has a header row and that the first column contains the image path and all the subsequent
    columns contain the target values which are bundled together into a numpy array.
    """
    image_targets = {}
    image_classes = {}
    with csv_path.open('r') as f:
        lines = f.readlines()
        start_line = 0
        # If there is a header, skip the first line
        if header:
            header_line = lines[0].strip().split(',')
            print(f'Header line of csv {csv_path} : {header_line}')
            start_line = 1
        for line in lines[start_line:]:
            line = line.strip().split(',')
            image_path = line[0]
            image_targets[image_path] = np.array([float(x) for x in line[2:]], dtype=np.float32)
            image_classes[image_path] = int(line[1])
    return image_targets,  image_classes


class RegressionImageFolder(datasets.ImageFolder):
    """
    The regression image folder is a subclass of the ImageFolder class and is designed for 
    image regression tasks rather than image classification tasks. It takes in dictionaries
    that map image paths to their target values and classes.
    """
    def __init__(
        self, root: str, image_targets: Dict[str, Any], image_classes: Dict[str, int], *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(root, *args, **kwargs)
        paths, _ = zip(*self.imgs)
        self.targets = np.array([image_targets[str(path)] for path in paths], dtype=np.float32)
        self.classes = np.array([image_classes[str(path)] for path in paths], dtype=np.int32)
        self.samples = self.imgs = list(zip(paths, self.targets, self.classes))

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Override the __getitem__ method to return image, target, and class.
        """
        path, target, class_ = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, class_

class RegressionTaskData:
    """
    This class is a wrapper for the data that is used in the regression task. It contains
    the train and test loaders.
    """
    def __init__(
        self,
        grayscale: bool = True,
        image_folder_path: Path = DEFAULT_IMAGE_FOLDER_PATH,
        batch_size: int = 64,
    ) -> None:
        self.grayscale = grayscale
        self.batch_size = batch_size
        self.image_folder_path = image_folder_path
        self.train_transforms, self.test_transforms, self.valid_transforms = get_transforms(grayscale)
        self.trainloader = self.make_trainloader(batch_size)
        self.testloader = self.make_testloader(batch_size)
        self.validloader = self.make_validloader(batch_size)

    @property
    def output_image_size(self):
        return (1 if self.grayscale else 3)

    def make_trainloader(self, batch_size: int = 64) -> torch.utils.data.DataLoader:
        """
        Builds the train data loader
        """
        image_targets, image_classes = load_image_targets_from_csv(self.image_folder_path / 'sa_train.csv')
        train_data = RegressionImageFolder(
            str(self.image_folder_path / 'train'), 
            image_targets, image_classes,
            transform=self.train_transforms
        )
        train_sampler = SubsetRandomSampler(torch.randperm(len(train_data))[:65536])
        
        # This constructs the dataloader that actually determines how images will be loaded in batches
        trainloader = torch.utils.data.DataLoader(train_data, batch_size,
                                                  drop_last=True,
                                                  #shuffle=True,
                                                  sampler=train_sampler
                                                  )
        return trainloader

    def make_testloader(self, batch_size: int = 64) -> torch.utils.data.DataLoader:
        """
        Builds the test data loader
        """
        image_targets, image_classes = load_image_targets_from_csv(self.image_folder_path / 'sa_test.csv')
        test_data = RegressionImageFolder(
            str(self.image_folder_path / 'test'), 
            image_targets, image_classes,
            transform=self.test_transforms
        )
        #test_sampler = SubsetRandomSampler(torch.randperm(len(test_data))[:1024])
        # This constructs the dataloader that actually determines how images will be loaded in batches
        testloader = torch.utils.data.DataLoader(test_data, batch_size,
                                                 #drop_last=True,
                                                 shuffle=True,
                                                 #sampler=test_sampler
                                                 )
        return testloader

    def make_validloader(self, batch_size: int = 64) -> torch.utils.data.DataLoader:
        """
        Builds the valid data loader
        """
        image_targets, image_classes = load_image_targets_from_csv(self.image_folder_path / 'sa_valid.csv')
        valid_data = RegressionImageFolder(
            str(self.image_folder_path / 'valid'), 
            image_targets, image_classes,
            transform=self.valid_transforms
        )
        #valid_sampler = SubsetRandomSampler(torch.randperm(len(valid_data))[:1024])
        # This constructs the dataloader that actually determines how images will be loaded in batches
        validloader = torch.utils.data.DataLoader(valid_data, batch_size,
                                                  #drop_last=True,
                                                  shuffle=True,
                                                  #sampler=valid_sampler
                                                  )
        return validloader

    def visualise_image(self):
        """
        This function visualises a single image from the train set
        """
        images, targets, classes = next(iter(self.testloader))
        print(targets[0].shape)
        print(images.shape)
        print(targets[0])
        print(classes[0])
        if self.grayscale:
            plt.imshow(images[0][0, :, :], cmap='gray')
        else:
            plt.imshow(images[0].permute(1, 2, 0))
        plt.show()

if __name__ == '__main__':
    data = RegressionTaskData()
    data.visualise_image()
