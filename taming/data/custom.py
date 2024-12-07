import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example



class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)

class CubTrain(CustomBase):
    def __init__(self,size,training_images_list_file,train_val_file,root="./data/CUB_200_2011/images"):
        super().__init__()
        paths=[]
        train_test_list=[]

        with open (train_val_file,"r") as f:
            for line in f:
                train_test_list.append(int(line[:-1].split(' ')[-1]))

        with open( training_images_list_file, "r") as f:
            for line in f:
                words = line.split()
                if len(words) > 1:
                    full_path = os.path.join(root, words[1])  # Combine root with the file path
                    paths.append(full_path)
        paths = [x for i, x in zip(train_test_list, paths) if i]
        self.data=ImagePaths(paths=paths,size=size,random_crop=True)



class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)

class CubTest(CustomBase):
    def __init__(self, size, test_images_list_file, train_val_file, root="./data/CUB_200_2011/images"):
        super().__init__()

        # Initialize empty lists for paths and train/test labels
        paths = []
        train_test_list = []
        with open(train_val_file,"r") as f:
        # Read train/test labels from train_val_file
            for line in f:
                train_test_list.append(int(line.strip().split(' ')[-1]))

        # Read test images and build paths
        with open(test_images_list_file, "r") as f:
            for line in f:
                words = line.split()
                if len(words) > 1:
                    full_path = os.path.join(root, words[1])
                    paths.append(full_path)

        # Filter paths based on train/test labels
        paths = [x for i, x in zip(train_test_list, paths) if not i]

        # Initialize the data attribute
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)

        # size 256*256

class CubTest_O(CustomBase):
    def __init__(self, size, test_images_list_file, train_val_file, root="./data/CUB_200_2011/images"):
        super().__init__()

        # Initialize empty lists for paths and train/test labels
        paths = []
        train_test_list = []
        with open(train_val_file,"r") as f:
        # Read train/test labels from train_val_file
            for line in f:
                train_test_list.append(int(line.strip().split(' ')[-1]))

        # Read test images and build paths
        with open(test_images_list_file, "r") as f:
            for line in f:
                words = line.split()
                if len(words) > 1:
                    full_path = os.path.join(root, words[1])
                    paths.append(full_path)

        # Filter paths based on train/test labels
        paths = [x for i, x in zip(train_test_list, paths) if not i]

        # Initialize the data attribute
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)

