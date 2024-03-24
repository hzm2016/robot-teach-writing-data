import glob
import random
import os
import cv2

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/D' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class CharDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))

    def __getitem__(self, index):

        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        return {'A': item_A, 'label': torch.tensor(index)}

    def __len__(self):
        return len(self.files_A)


if __name__ == '__main__':

    dir_root = 'datasets/seq/train'
    img_dirs_part = os.path.join(dir_root,'imgs_ske')
    print(img_dirs_part)
    for folder_name in glob.glob(img_dirs_part+'/*'):
        for img_name in sorted(glob.glob(folder_name + '/*.jpg'))[1:]:
            img = cv2.imread(img_name)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            succ = cv2.imwrite(img_name, img[:,:,0])

