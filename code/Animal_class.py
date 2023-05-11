import os
import pickle
import numpy as np
from torch.utils.data import Dataset
import torch
from torchvision.transforms import ToTensor, ToPILImage, Compose, Resize
from torchvision.datasets import CIFAR10, StanfordCars
import cv2
from PIL import Image


class AnimalDataset(Dataset):
    def __init__(self,
                 root,
                 train=True,
                 transform=None):
        self.transform = transform
        animal_path = os.path.join(root, 'data', 'animals')

        if train:
            self.data_path = [os.path.join(animal_path, f) for f in os.listdir(animal_path) if f == 'train'][0]
        else:
            self.data_path = [os.path.join(animal_path, f) for f in os.listdir(animal_path) if f == 'test'][0]

        self.images = []
        self.labels = []
        self.categories = [fol for fol in os.listdir(self.data_path)]
        for i, folder in enumerate(os.listdir(self.data_path)):
            for file in os.listdir(os.path.join(self.data_path, folder)):
                if os.path.splitext(file)[-1].lower() in ['.jpg', '.jpeg', '.png']:
                    image_file = os.path.join(self.data_path, folder, file)
                    self.images.append(image_file)
                    self.labels.append(i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB') #TODO: why image has 4 chanels
        # Transform to tensor
        if self.transform:
            img = self.transform(img)

        # Get label
        label = self.labels[idx]
        return img, label


if __name__ == '__main__':
    # Transform
    transform = Compose([
        Resize((500, 500)),
        ToTensor() # transform to Tensor with range from 0 to 1
    ])

    root = '../'
    dataset = AnimalDataset(root=root, train=True, transform=transform)
    index = 4000
    image, label_ = dataset.__getitem__(index)

    print(dataset.__len__())

    print(image.shape)
    print(label_)
    print(dataset.categories)
    transform_PIL = ToPILImage()
    PIL_image = transform_PIL(image)
    PIL_image.show()
