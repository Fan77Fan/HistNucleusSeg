import h5py
import os
import numpy
import math
import threading
import glob
from scipy import misc
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset


class DataPartition(object):
    '''
    partition the dataset into folds for cross validation
    '''
    def __init__(self, n_fold=5, val_ratio=0.25, seed=905, path=None):
        self.n_fold = n_fold
        self.val_ratio = val_ratio
        self.seed = seed
        self.path = path
        self.filename_mat = self.parse_filenames()
        self.idx_list = self.generate_index()

    def parse_filenames(self):
        filepath = self.path
        file_list = sorted(glob.glob(os.path.join(filepath, 'Slide_*')))
        gt_list = sorted(glob.glob(os.path.join(filepath, 'GT_*')))
        n_file = len(file_list)
        filename_mat = numpy.zeros(shape=(n_file, 3), dtype=object)
        for i, filename in enumerate(file_list):
            filename_mat[i, :] = [i, filename[len(filepath)+1:], gt_list[i][len(filepath)+1:]]
        return filename_mat

    def generate_index(self):
        numpy.random.seed(self.seed)
        idx = self.filename_mat[:, 0].astype(int)
        numpy.random.shuffle(idx)
        idx_fold = self.split_seq(idx, size=self.n_fold)
        idx_list = []
        for i in range(0, self.n_fold):
            idx_tes = idx_fold[i]
            temp = [x for x in idx if x not in idx_tes]
            numpy.random.shuffle(temp)
            n_nontest = len(temp)
            n_val = round(n_nontest * self.val_ratio)
            idx_val = temp[0:n_val]
            idx_tra = temp[n_val:]
            index_dict = {'train': sorted(list(idx_tra)),
                          'valid': sorted(list(idx_val)),
                          'test':  sorted(list(idx_tes))}
            idx_list.append(index_dict)
        return idx_list

    def fold_idx(self, i_fold, verbose=True):
        '''
        output index information given a certain fold number
        :param i_fold: the number of fold
        :param verbose: whether to print partition information
        :return: a dictionary describing fold index information
        '''
        idx_dict = self.idx_list[i_fold]
        idx_train = idx_dict['train']
        idx_valid = idx_dict['valid']
        idx_test = idx_dict['test']
        if verbose is True:
            print('fold: ' + str(i_fold))
            print('  train: ', idx_train)
            print('  valid: ', idx_valid)
            print('  test : ', idx_test)
        partinfo = {'train': self.filename_mat[idx_train, :],
                    'valid': self.filename_mat[idx_valid, :],
                    'test': self.filename_mat[idx_test, :]}
        return partinfo

    def all_idx(self, verbose=False):
        idx_all = self.filename_mat[:, 0].astype(int)
        idx_all = sorted(list(idx_all))
        if verbose is True:
            print('all index: ', idx_all)
        idxinfo = {'all': self.filename_mat[idx_all, :]}
        return idxinfo

    @staticmethod
    def split_seq(seq, size):
        newseq = []
        splitsize = 1.0 / size * len(seq)
        for i in range(size):
            newseq.append(seq[int(round(i * splitsize)):int(round((i + 1) * splitsize))])
        return newseq


class ImageDataset(Dataset):
    '''
    extract image data using the index information generated from DataPartition.fold_idx()
    '''
    def __init__(self, index_info, sourcedir):
        self.sourcedir = sourcedir
        self.index_info = index_info
        self.dataset = self.extract_dataset()
        # self.padding = padding

    def __len__(self):
        return self.dataset['image'].shape[0]

    def __getitem__(self, item):
        '''
        extract certain image from the dataset
        :param item: index of image
        :return: dictionary of image and its segmentation map
        '''
        image = self.dataset['image'][item, :, :, :]
        seg = self.dataset['seg'][item, :, :, :]
        data = {'image': image, 'seg': seg, 'image_size': seg.shape[1:3]}

        return data

    def extract_dataset(self):
        image_list = []
        gt_list = []
        # fileinfo_list = []
        for i, imagefolder, gtfolder in self.index_info:
            for image_path in glob.glob(os.path.join(self.sourcedir, imagefolder, '*.png')):
                image = misc.imread(image_path)
                # print(image.shape[2])  # some of the png images only have 3 channels..tricky!
                if image.shape[2] > 3:
                    image_list.append(image[:, :, 0:-1])
                elif image.shape[2] == 3:
                    image_list.append(image[:, :, :])
            for gt_path in glob.glob(os.path.join(self.sourcedir, gtfolder, '*.png')):
                gt = misc.imread(gt_path)
                gt[gt > 0] = 1
                gt_list.append(numpy.stack((gt, numpy.ones_like(gt) - gt), axis=2))
                # fileinfo_list.append((i, gt_path))
        image_mat = numpy.stack(image_list, axis=0)
        gt_mat = numpy.stack(gt_list, axis=0)
        # print(gt_mat.shape)
        return {'image': image_mat, 'seg': gt_mat}

