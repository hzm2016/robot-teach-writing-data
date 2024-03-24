import glob
import random
import os
import cv2
import torch
import copy

from torch.utils.data import Dataset
from sklearn.metrics import pairwise_distances
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

class SequentialImageDataset(Dataset):
    def __init__(self, root, unaligned=False, mode='train'):
        self.unaligned = unaligned
        self.data_list = []
        self.normalize = True

        dir_root = root + '/' + mode
        img_dirs_part_points = os.path.join(dir_root, 'imgs_part_points')
        self._form_dataset(img_dirs_part_points)

    @staticmethod
    def rotate(points, angle=15):

        x_mean = points[:, 0].mean()
        y_mean = points[:, 1].mean()

        origin = [x_mean, y_mean]

        # +- 30 degrees
        angle = random.uniform(-1, 1) * np.pi * angle / 180

        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])

        o = np.atleast_2d(origin)
        p = np.atleast_2d(points)
        return np.squeeze((R @ (p.T-o.T) + o.T).T)

    @staticmethod
    def scale(points, range=0.3):

        x_mean = points[:, 0].mean()
        y_mean = points[:, 1].mean()

        origin = [x_mean, y_mean]

        distance = points - origin

        scale_factor = random.uniform(1-range, 1 + range)
        scaled_distance = distance * scale_factor

        scaled_points = origin + scaled_distance

        return np.clip(scaled_points, 0, 127)

    @staticmethod
    def shift(points, max_shift=10):

        n_points = copy.deepcopy(points)
        x_shift_value = random.randrange(-max_shift, max_shift)
        y_shift_value = random.randrange(-max_shift, max_shift)

        n_points[:, 0] = points[:, 0] + x_shift_value
        n_points[:, 1] = points[:, 1] + y_shift_value

        return n_points

    def cut(self, points, ratio=0.1):

        if points.shape[0] < 5:
            return points
            
        if random.uniform(0, 1) > 0.5:
            rt_points = points[int(ratio * points.shape[0]):]
        else:
            rt_points = points[:-int(ratio * points.shape[0])]

        return rt_points

    def drop(self, points, ratio=0.1):
        
        if points.shape[0] < 5:
            return points

        index = random.sample(range(points.shape[0]), int(
            points.shape[0] * (1-ratio)))

        return points[sorted(index)]

    def _rank_file_accd_num(self, all_files):

        num_list = []
        for file_name in all_files:
            num_list.append(int(file_name.split('/')[-1].split('.')[0]))

        sorted_files = []
        sort_index = np.argsort(num_list)

        for i in sort_index:
            sorted_files.append(all_files[i])
        return sorted_files

    def aug_points(self, points):

        func_list = [self.rotate, self.shift, self.scale, self.drop, self.cut]
        func_index = random.randint(0, 4)

        m_points = func_list[func_index](points)

        return m_points

    def group_aug_points(self, points):

        func_list = [self.rotate, self.shift, self.scale]
        func_index = random.randint(1, 1)

        # for func_index in range(3):
        points = func_list[func_index](points)
        return points

    def form_train_sample(self, file_list):

        points_list = []
        modified_points_list = []
        num_list = []
        modified_num_list = []
        self.determinstic = True

        for file_name in file_list[:-1]:
            points = np.loadtxt(file_name)
            points_list.append(points)
            num_list.append(points.shape[0])
            if not self.determinstic:
                aug_points = self.aug_points(points)
                modified_num_list.append(aug_points.shape[0])
                modified_points_list.append(aug_points)

        cur_points = np.loadtxt(file_list[-1])
        edge_index_c = pairwise_distances(cur_points) < 12.8
        edge_index_c = edge_index_c.astype(int)

        points_list = np.concatenate(points_list)
        edge_index_ori = pairwise_distances(points_list) < 12.8
        edge_index_ori = edge_index_ori.astype(int)

        # For deterministic model
        c_points = copy.deepcopy(cur_points)
        if self.determinstic:
            modified_points_list = self.group_aug_points(
                np.concatenate([points_list, c_points]))
            last_index = c_points.shape[0]
            label_points = modified_points_list[-last_index:, :]
            modified_points_list = modified_points_list[:-last_index, :]
        else:
            modified_points_list = np.concatenate(modified_points_list)

        edge_index_modified = pairwise_distances(modified_points_list) < 12.8
        edge_index_modified = edge_index_modified.astype(int)

        num_list = np.array(num_list)
        if len(modified_num_list) > 0:
            modified_num_list = np.array(modified_num_list)
        else:
            modified_num_list = np.array(num_list)

        dist_dis_x = wasserstein_distance(modified_points_list[:,0], points_list[:,0])
        dist_dis_y = wasserstein_distance(modified_points_list[:,1], points_list[:,1])

        # print(modified_points_list- points_list)
        if self.normalize:
            modified_points_list = modified_points_list / 128.0
            points_list = points_list / 128.0
            cur_points = cur_points / 128.0
            dist_dis_x = dist_dis_x / 128.0
            dist_dis_y = dist_dis_y / 128.0
        

        rt_dict = {
            'm_points': torch.from_numpy(modified_points_list),
            'o_points': torch.from_numpy(points_list),
            'c_points': torch.from_numpy(cur_points),
            'edge_index_o': torch.from_numpy(edge_index_ori),
            'edge_index_m': torch.from_numpy(edge_index_modified),
            'edge_index_c': torch.from_numpy(edge_index_c),
            'dist_dis': torch.Tensor([dist_dis_x,dist_dis_y]),
            'num': num_list,
            'm_num': modified_num_list
        }

        if self.determinstic and self.normalize:
            rt_dict.update({'l_points': torch.from_numpy(label_points/128.0)})
        elif self.determinstic:
            rt_dict.update({'l_points': torch.from_numpy(label_points)})

        return rt_dict

    def _form_dataset(self, path):

        for folder_name in glob.glob(path+'/*'):
            file_list = []
            length = len(self._rank_file_accd_num(
                glob.glob(folder_name + '/*.txt')))
            if length < 3:
                continue
            for idx, txt_name in enumerate(self._rank_file_accd_num(glob.glob(folder_name + '/*.txt'))):
                file_list.append(txt_name)
                if idx < 8:
                    continue
                self.data_list.append(file_list)

    def __getitem__(self, index):
        return self.form_train_sample(self.data_list[index])

    def __len__(self):
        return len(self.data_list)


def visulize_points(points, name):

    img_canvas = np.full((128, 128), 255, np.uint8)
    points = [points.astype(int)]
    for l in points:
        c = (0, 0, 0)
        for i in range(0, len(l)-1):
            cv2.line(img_canvas, (l[i][0], l[i][1]),
                     (l[i+1][0], l[i+1][1]), c, 2)

    cv2.imwrite(name, img_canvas)


def unit_test_1():

    filename = '/home/cunjun/Robot-Teaching-Assiantant/gan/data/seq/train/imgs_part_points/且/0.txt'
    points = np.loadtxt(open(filename, 'r'))
    visulize_points(points, 'ori.jpg')
    points = SequentialImageDataset.shift(points)
    visulize_points(points, 'new.jpg')


def unit_test_2():

    root = '/home/cunjun/Robot-Teaching-Assiantant/gan/data/seq'
    SequentialImageDataset(root)


if __name__ == '__main__':

    unit_test_1()
    unit_test_2()

    # dir_root = 'datasets/seq/train'
    # img_dirs_part = os.path.join(dir_root,'imgs_ske')
    # for folder_name in glob.glob(img_dirs_part+'/*'):
    #     for img_name in sorted(glob.glob(folder_name + '/*.jpg'))[1:]:
    #         img = cv2.imread(img_name)
    #         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         succ = cv2.imwrite(img_name, img[:,:,0])
