import numpy
import os
import glob
import time
import nibabel
import sys
import h5py

from setup_variables import data_path
from setup_dataIO import DataPartition, ImageDataset
from setup_utils import add_image_feature, add_contectual_feature

# ===activate when using OSX===
import matplotlib
# matplotlib.use('TkAgg')
# matplotlib.use('Agg')

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def main():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print('device: ' + str(device))
    numpy.set_printoptions(precision=3)
    numpy.random.seed(6372)
    window_width = 4

    dataIO = DataPartition(path=data_path)

    part_info = dataIO.fold_idx(i_fold=0, verbose=True)

    image_dataset_tra = ImageDataset(index_info=part_info['train'], sourcedir=data_path)
    image_dataset_val = ImageDataset(index_info=part_info['valid'], sourcedir=data_path)
    image_dataset_tes = ImageDataset(index_info=part_info['test'], sourcedir=data_path)

    Y_tra = image_dataset_tra.dataset['seg'].reshape((-1, 2))[:, 0]
    Y_val = image_dataset_val.dataset['seg'].reshape((-1, 2))[:, 0]
    Y_tes = image_dataset_tes.dataset['seg'].reshape((-1, 2))[:, 0]

    image_size = image_dataset_tra.dataset['image'].shape[1:3]
    n_image_tra = image_dataset_tra.dataset['image'].shape[0]
    n_image_val = image_dataset_val.dataset['image'].shape[0]
    n_image_tes = image_dataset_tes.dataset['image'].shape[0]

    image_tra_mean = image_dataset_tra.dataset['image'].mean(3)
    image_val_mean = image_dataset_val.dataset['image'].mean(3)
    image_tes_mean = image_dataset_tes.dataset['image'].mean(3)

    print('\n')

    # -----------------------------------------------------------------
    print('phase I training.')
    X_tra_1 = image_dataset_tra.dataset['image'].reshape((-1, 3))

    temp_all_idx = numpy.arange(0, Y_tra.shape[0])  # select only a subset of training set for training
    temp_idx_select_pos = numpy.random.choice(temp_all_idx[Y_tra > 0], size=10000, replace=True)
    temp_idx_select_neg = numpy.random.choice(temp_all_idx[Y_tra < 1], size=10000, replace=True)
    idx_select_1 = numpy.sort(numpy.concatenate((temp_idx_select_pos, temp_idx_select_neg)))

    clf_1 = RandomForestClassifier(n_estimators=20, min_samples_leaf=10)
    clf_1.fit(X_tra_1[idx_select_1, :], Y_tra[idx_select_1])

    X_val_1 = image_dataset_val.dataset['image'].reshape((-1, 3))

    print('  predicting validation set.')
    temp_score_val_pred = clf_1.predict_proba(X_val_1)[:, 1]
    metric_auc_1 = roc_auc_score(y_true=Y_val, y_score=temp_score_val_pred)
    print('  AUC = {:.3f}\n'.format(metric_auc_1))

    # temp_ind_tes = 2  # demo
    # temp_X_tes = image_dataset_tes[temp_ind_tes]['image'].reshape((-1, 3))
    # image_size = image_dataset_tra[temp_ind_tes]['image_size']
    #
    # temp_score_tes_pred = clf.predict_proba(temp_X_tes)[:, 1]
    # temp_Y_tes_pred = numpy.zeros_like(temp_score_tes_pred)
    # temp_Y_tes_pred[temp_score_tes_pred >= 0.5] = 1
    #
    # temp_seg_pred = temp_Y_tes_pred.reshape(image_size)
    #
    # plt.figure(figsize=(3 * window_width, 1 * window_width))
    # plt.subplot(1, 3, 1)
    # plt.imshow(image_dataset_tes[temp_ind_tes]['image'][:, :, 0], cmap='gray')
    # plt.subplot(1, 3, 2)
    # plt.imshow(image_dataset_tes[temp_ind_tes]['seg'][:, :, 0], cmap='gray')
    # plt.subplot(1, 3, 3)
    # plt.imshow(temp_seg_pred, cmap='gray')
    # plt.show()

    # -----------------------------------------------------------------
    # print('phase I training with more features.')
    #
    # image_tra = image_dataset_tra.dataset['image']
    # image_tra_aug = add_image_feature(image_set=image_tra)
    # image_tra_aug = numpy.concatenate((image_tra, image_tra_aug), axis=3)
    # n_feature = image_tra_aug.shape[3]
    # X_tra = image_tra_aug.reshape((-1, n_feature))
    #
    # clf = RandomForestClassifier(n_estimators=20, min_samples_leaf=10)
    # clf.fit(X_tra[idx_select, :], Y_tra[idx_select])
    #
    # image_tes = image_dataset_tes.dataset['image']
    # image_tes_aug = add_image_feature(image_set=image_tes)
    # image_tes_aug = numpy.concatenate((image_tes, image_tes_aug), axis=3)
    # X_tes = image_tes_aug.reshape((-1, n_feature))
    #
    # print('  predicting.')
    # score_tes_pred = clf.predict_proba(X_tes)[:, 1]
    # metric_auc = roc_auc_score(y_true=Y_tes, y_score=score_tes_pred)
    # print('  AUC = {:.3f}\n'.format(metric_auc))

    # -----------------------------------------------------------------
    print('phase II training.')

    # score_tra_pred_1 = clf_1.predict_proba(X_tra_1)[:, 1]
    score_tra_pred_mat_1 = clf_1.predict_proba(X_tra_1)[:, 1].reshape((n_image_tra, ) + image_size)
    context_tra_2 = add_contectual_feature(image_set=score_tra_pred_mat_1)
    imgfeature_tra_2 = add_contectual_feature(image_set=image_tra_mean)
    X_tra_2 = numpy.concatenate((X_tra_1, context_tra_2.reshape((-1, 8)), imgfeature_tra_2.reshape((-1, 3))), axis=1)

    temp_all_idx = numpy.arange(0, Y_tra.shape[0])  # select only a subset of training set for training
    temp_idx_select_pos = numpy.random.choice(temp_all_idx[Y_tra > 0], size=10000, replace=True)
    temp_idx_select_neg = numpy.random.choice(temp_all_idx[Y_tra < 1], size=10000, replace=True)
    idx_select_2 = numpy.sort(numpy.concatenate((temp_idx_select_pos, temp_idx_select_neg)))

    clf_2 = RandomForestClassifier(n_estimators=20, min_samples_leaf=10)
    clf_2.fit(X_tra_2[idx_select_2, :], Y_tra[idx_select_2])

    # score_val_pred = clf_1.predict_proba(X_val_1)[:, 1]
    score_val_pred_mat_1 = clf_1.predict_proba(X_val_1)[:, 1].reshape((n_image_val,) + image_size)
    context_val_2 = add_contectual_feature(image_set=score_val_pred_mat_1)
    imgfeature_val_2 = add_image_feature(image_set=score_val_pred_mat_1)
    X_val_2 = numpy.concatenate((X_val_1, context_val_2.reshape((-1, 8)), imgfeature_val_2.reshape((-1, 3))), axis=1)

    print('  predicting validation set.')
    temp_score_val_pred_2 = clf_2.predict_proba(X_val_2)[:, 1]
    metric_auc_2 = roc_auc_score(y_true=Y_val, y_score=temp_score_val_pred_2)
    print('  AUC = {:.3f}\n'.format(metric_auc_2))


    # -----------------------------------------------------------------
    print('phase III training.')

    score_tra_pred_mat_2 = clf_2.predict_proba(X_tra_2)[:, 1].reshape((n_image_tra, ) + image_size)
    context_tra_3 = add_contectual_feature(image_set=score_tra_pred_mat_2)
    X_tra_3 = numpy.concatenate((X_tra_1, context_tra_3.reshape((-1, 8))), axis=1)

    temp_all_idx = numpy.arange(0, Y_tra.shape[0])  # select only a subset of training set for training
    temp_idx_select_pos = numpy.random.choice(temp_all_idx[Y_tra > 0], size=10000, replace=True)
    temp_idx_select_neg = numpy.random.choice(temp_all_idx[Y_tra < 1], size=10000, replace=True)
    idx_select_3 = numpy.sort(numpy.concatenate((temp_idx_select_pos, temp_idx_select_neg)))

    clf_3 = RandomForestClassifier(n_estimators=20, min_samples_leaf=10)
    clf_3.fit(X_tra_3[idx_select_3, :], Y_tra[idx_select_3])

    score_val_pred_mat_2 = clf_2.predict_proba(X_val_2)[:, 1].reshape((n_image_val,) + image_size)
    context_val_3 = add_contectual_feature(image_set=score_val_pred_mat_2)
    X_val_3 = numpy.concatenate((X_val_1, context_val_3.reshape((-1, 8))), axis=1)

    print('  predicting validation set.')
    temp_score_val_pred_3 = clf_3.predict_proba(X_val_3)[:, 1]
    metric_auc_3 = roc_auc_score(y_true=Y_val, y_score=temp_score_val_pred_3)
    print('  AUC = {:.3f}\n'.format(metric_auc_3))

    # -----------------------------------------------------------------
    print('phase IV training.')

    score_tra_pred_mat_3 = clf_3.predict_proba(X_tra_3)[:, 1].reshape((n_image_tra,) + image_size)
    context_tra_4 = add_contectual_feature(image_set=score_tra_pred_mat_3)
    X_tra_4 = numpy.concatenate((X_tra_1, context_tra_4.reshape((-1, 8))), axis=1)

    temp_all_idx = numpy.arange(0, Y_tra.shape[0])  # select only a subset of training set for training
    temp_idx_select_pos = numpy.random.choice(temp_all_idx[Y_tra > 0], size=10000, replace=True)
    temp_idx_select_neg = numpy.random.choice(temp_all_idx[Y_tra < 1], size=10000, replace=True)
    idx_select_4 = numpy.sort(numpy.concatenate((temp_idx_select_pos, temp_idx_select_neg)))

    clf_4 = RandomForestClassifier(n_estimators=20, min_samples_leaf=10)
    clf_4.fit(X_tra_4[idx_select_4, :], Y_tra[idx_select_4])

    score_val_pred_mat_3 = clf_3.predict_proba(X_val_3)[:, 1].reshape((n_image_val,) + image_size)
    context_val_4 = add_contectual_feature(image_set=score_val_pred_mat_3)
    X_val_4 = numpy.concatenate((X_val_1, context_val_4.reshape((-1, 8))), axis=1)

    print('  predicting validation set.')
    temp_score_val_pred_4 = clf_4.predict_proba(X_val_4)[:, 1]
    metric_auc_4 = roc_auc_score(y_true=Y_val, y_score=temp_score_val_pred_4)
    print('  AUC = {:.3f}\n'.format(metric_auc_4))

    print('pause')


if __name__ == '__main__':
    main()

