import numpy
import os
import glob
import time
import nibabel
import sys
import h5py

from setup_variables import data_path
from setup_dataIO import DataPartition, ImageDataset
from setup_utils import add_image_feature, add_contectual_feature, preprocess_image, process_score_sp

# ===activate when using OSX===
import matplotlib
# matplotlib.use('TkAgg')
# matplotlib.use('Agg')

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def main():
    numpy.set_printoptions(precision=3)
    numpy.random.seed(6372)
    window_width = 4

    dataIO = DataPartition(path=data_path, val_ratio=0)

    for i_fold in (0, 1, 2, 3, 4):
        part_info = dataIO.fold_idx(i_fold=i_fold, verbose=True)

        image_dataset_tra = ImageDataset(index_info=part_info['train'], sourcedir=data_path)
        image_dataset_tes = ImageDataset(index_info=part_info['test'], sourcedir=data_path)

        Y_tra = image_dataset_tra.dataset['seg'].reshape((-1, 2))[:, 0]
        Y_tes = image_dataset_tes.dataset['seg'].reshape((-1, 2))[:, 0]

        image_size = image_dataset_tra.dataset['image'].shape[1:3]

        n_image_tra = image_dataset_tra.dataset['image'].shape[0]
        n_image_tes = image_dataset_tes.dataset['image'].shape[0]

        image_tra_mean = image_dataset_tra.dataset['image'].mean(3)
        image_tes_mean = image_dataset_tes.dataset['image'].mean(3)

        image_tra_fea = add_image_feature(image_set=image_tra_mean)
        image_tes_fea = add_image_feature(image_set=image_tes_mean)

        image_tra_pre, sp_tra = preprocess_image(image_set=image_dataset_tra.dataset['image'])
        image_tes_pre, sp_tes = preprocess_image(image_set=image_dataset_tes.dataset['image'])

        X_tra = image_dataset_tra.dataset['image'].reshape((-1, 3))
        X_tes = image_dataset_tes.dataset['image'].reshape((-1, 3))

        # -----------------------------------------------------------------
        print('phase I training.')
        X_tra_1 = numpy.concatenate((X_tra, image_tra_pre.reshape(-1, 3), image_tra_fea.reshape(-1, 3)), axis=1)

        temp_all_idx = numpy.arange(0, Y_tra.shape[0])  # select only a subset of training set for training
        temp_idx_select_pos = numpy.random.choice(temp_all_idx[Y_tra > 0], size=10000, replace=True)
        temp_idx_select_neg = numpy.random.choice(temp_all_idx[Y_tra < 1], size=10000, replace=True)
        idx_select_1 = numpy.sort(numpy.concatenate((temp_idx_select_pos, temp_idx_select_neg)))

        clf_1 = RandomForestClassifier(n_estimators=20, min_samples_leaf=200, max_depth=None, max_features='auto')
        clf_1.fit(X_tra_1[idx_select_1, :], Y_tra[idx_select_1])

        X_tes_1 = numpy.concatenate((X_tes, image_tes_pre.reshape(-1, 3), image_tes_fea.reshape(-1, 3)), axis=1)

        print('  predicting training set.', end='')
        score_tra_pred_1 = clf_1.predict_proba(X_tra_1)[:, 1]
        metric_auc = roc_auc_score(y_true=Y_tra, y_score=score_tra_pred_1)
        print('  AUC = {:.3f}\n'.format(metric_auc))

        # print('  predicting test set.', end='')
        # temp_score_tes_pred = clf_1.predict_proba(X_tes_1)[:, 1]
        # metric_auc = roc_auc_score(y_true=Y_tes, y_score=temp_score_tes_pred)
        # print('  AUC = {:.3f}'.format(metric_auc))


        # -----------------------------------------------------------------
        print('phase II training.')

        d = 3
        score_tra_pred_mat_1 = score_tra_pred_1.reshape((n_image_tra, ) + image_size)
        context_tra_2 = add_contectual_feature(image_set=score_tra_pred_mat_1, d=d)
        superscore_tra_2 = process_score_sp(image_set=score_tra_pred_mat_1, sp_set=sp_tra)
        imgfeature_tra_2 = add_contectual_feature(image_set=image_tra_mean, d=d)
        X_tra_2 = numpy.concatenate((X_tra_1, context_tra_2.reshape(-1, 8), imgfeature_tra_2.reshape(-1, 8), superscore_tra_2.reshape(-1, 2)), axis=1)

        temp_all_idx = numpy.arange(0, Y_tra.shape[0])  # select only a subset of training set for training
        temp_idx_select_pos = numpy.random.choice(temp_all_idx[Y_tra > 0], size=10000, replace=True)
        temp_idx_select_neg = numpy.random.choice(temp_all_idx[Y_tra < 1], size=10000, replace=True)
        idx_select_2 = numpy.sort(numpy.concatenate((temp_idx_select_pos, temp_idx_select_neg)))

        clf_2 = RandomForestClassifier(n_estimators=20, min_samples_leaf=200)
        clf_2.fit(X_tra_2[idx_select_2, :], Y_tra[idx_select_2])

        score_tes_pred_mat_1 = clf_1.predict_proba(X_tes_1)[:, 1].reshape((n_image_tes,) + image_size)
        context_tes_2 = add_contectual_feature(image_set=score_tes_pred_mat_1, d=d)
        superscore_tes_2 = process_score_sp(image_set=score_tes_pred_mat_1, sp_set=sp_tes)
        imgfeature_tes_2 = add_contectual_feature(image_set=image_tes_mean, d=d)
        X_tes_2 = numpy.concatenate((X_tes_1, context_tes_2.reshape(-1, 8), imgfeature_tes_2.reshape(-1, 8), superscore_tes_2.reshape(-1, 2)), axis=1)

        print('  predicting training set.', end='')
        score_tra_pred_2 = clf_2.predict_proba(X_tra_2)[:, 1]
        metric_auc = roc_auc_score(y_true=Y_tra, y_score=score_tra_pred_2)
        print('  AUC = {:.3f}\n'.format(metric_auc))

        # print('  test set.', end='')
        # temp_score_tes_pred = clf_2.predict_proba(X_tes_2)[:, 1]
        # metric_auc = roc_auc_score(y_true=Y_tes, y_score=temp_score_tes_pred)
        # print('  AUC = {:.3f}'.format(metric_auc))


        # -----------------------------------------------------------------
        print('phase III training.')

        d = 6
        score_tra_pred_mat_2 = score_tra_pred_2.reshape((n_image_tra, ) + image_size)
        context_tra_3 = add_contectual_feature(image_set=score_tra_pred_mat_2, d=d)
        superscore_tra_3 = process_score_sp(image_set=score_tra_pred_mat_2, sp_set=sp_tra)
        imgfeature_tra_3 = add_contectual_feature(image_set=image_tra_mean, d=d)
        X_tra_3 = numpy.concatenate((X_tra_2, context_tra_3.reshape(-1, 8), imgfeature_tra_3.reshape(-1, 8), superscore_tra_3.reshape(-1, 2)), axis=1)

        temp_all_idx = numpy.arange(0, Y_tra.shape[0])  # select only a subset of training set for training
        temp_idx_select_pos = numpy.random.choice(temp_all_idx[Y_tra > 0], size=10000, replace=True)
        temp_idx_select_neg = numpy.random.choice(temp_all_idx[Y_tra < 1], size=10000, replace=True)
        idx_select_3 = numpy.sort(numpy.concatenate((temp_idx_select_pos, temp_idx_select_neg)))

        clf_3 = RandomForestClassifier(n_estimators=20, min_samples_leaf=20)
        clf_3.fit(X_tra_3[idx_select_3, :], Y_tra[idx_select_3])

        score_tes_pred_mat_2 = clf_2.predict_proba(X_tes_2)[:, 1].reshape((n_image_tes,) + image_size)
        context_tes_3 = add_contectual_feature(image_set=score_tes_pred_mat_2, d=d)
        superscore_tes_3 = process_score_sp(image_set=score_tes_pred_mat_2, sp_set=sp_tes)
        imgfeature_tes_3 = add_contectual_feature(image_set=image_tes_mean, d=d)
        X_tes_3 = numpy.concatenate((X_tes_2, context_tes_3.reshape(-1, 8), imgfeature_tes_3.reshape(-1, 8), superscore_tes_3.reshape(-1, 2)), axis=1)

        print('  predicting training set.', end='')
        temp_score_tra_pred = clf_3.predict_proba(X_tra_3)[:, 1]
        metric_auc = roc_auc_score(y_true=Y_tra, y_score=temp_score_tra_pred)
        print('  AUC = {:.3f}\n'.format(metric_auc))

        print('================')
        print('  test set.', end='')
        temp_score_tes_pred = clf_3.predict_proba(X_tes_3)[:, 1]
        metric_auc = roc_auc_score(y_true=Y_tes, y_score=temp_score_tes_pred)
        print('  AUC = {:.3f}\n\n'.format(metric_auc))


        # -----------------------------------------------------------------
        # print('phase IV training.')
        #
        # d = 8
        # score_tra_pred_mat_3 = clf_3.predict_proba(X_tra_3)[:, 1].reshape((n_image_tra,) + image_size)
        # context_tra_4 = add_contectual_feature(image_set=score_tra_pred_mat_3, d=d)
        # imgfeature_tra_4 = add_contectual_feature(image_set=image_tra_mean, d=d)
        # X_tra_4 = numpy.concatenate((X_tra_3, context_tra_4.reshape((-1, 8)), imgfeature_tra_4.reshape((-1, 8))), axis=1)
        #
        # temp_all_idx = numpy.arange(0, Y_tra.shape[0])  # select only a subset of training set for training
        # temp_idx_select_pos = numpy.random.choice(temp_all_idx[Y_tra > 0], size=10000, replace=True)
        # temp_idx_select_neg = numpy.random.choice(temp_all_idx[Y_tra < 1], size=10000, replace=True)
        # idx_select_4 = numpy.sort(numpy.concatenate((temp_idx_select_pos, temp_idx_select_neg)))
        #
        # clf_4 = RandomForestClassifier(n_estimators=20, min_samples_leaf=10)
        # clf_4.fit(X_tra_4[idx_select_4, :], Y_tra[idx_select_4])
        #
        # score_tes_pred_mat_3 = clf_3.predict_proba(X_tes_3)[:, 1].reshape((n_image_tes,) + image_size)
        # context_tes_4 = add_contectual_feature(image_set=score_tes_pred_mat_3, d=d)
        # imgfeature_tes_4 = add_contectual_feature(image_set=image_tes_mean, d=d)
        # X_tes_4 = numpy.concatenate((X_tes_3, context_tes_4.reshape((-1, 8)), imgfeature_tes_4.reshape((-1, 8))), axis=1)
        #
        # print('  predicting test set.', end='')
        # temp_score_tes_pred = clf_4.predict_proba(X_tes_4)[:, 1]
        # metric_auc = roc_auc_score(y_true=Y_tes, y_score=temp_score_tes_pred)
        # print('  AUC = {:.3f}'.format(metric_auc))
        #
        # print('  training set.', end='')
        # temp_score_tra_pred = clf_4.predict_proba(X_tra_4)[:, 1]
        # metric_auc = roc_auc_score(y_true=Y_tra, y_score=temp_score_tra_pred)
        # print('  AUC = {:.3f}\n'.format(metric_auc))


        # -----------------------------------------------------------------
        # print('phase V training.')
        #
        # d = 8
        # score_tra_pred_mat_4 = clf_4.predict_proba(X_tra_4)[:, 1].reshape((n_image_tra,) + image_size)
        # context_tra_5 = add_contectual_feature(image_set=score_tra_pred_mat_4, d=d)
        # imgfeature_tra_5 = add_contectual_feature(image_set=image_tra_mean, d=d)
        # X_tra_5 = numpy.concatenate((X_tra_4, context_tra_5.reshape((-1, 8)), imgfeature_tra_5.reshape((-1, 8))), axis=1)
        #
        # temp_all_idx = numpy.arange(0, Y_tra.shape[0])  # select only a subset of training set for training
        # temp_idx_select_pos = numpy.random.choice(temp_all_idx[Y_tra > 0], size=10000, replace=True)
        # temp_idx_select_neg = numpy.random.choice(temp_all_idx[Y_tra < 1], size=10000, replace=True)
        # idx_select_5 = numpy.sort(numpy.concatenate((temp_idx_select_pos, temp_idx_select_neg)))
        #
        # clf_5 = RandomForestClassifier(n_estimators=20, min_samples_leaf=10)
        # clf_5.fit(X_tra_5[idx_select_5, :], Y_tra[idx_select_5])
        #
        # score_tes_pred_mat_4 = clf_4.predict_proba(X_tes_4)[:, 1].reshape((n_image_tes,) + image_size)
        # context_tes_5 = add_contectual_feature(image_set=score_tes_pred_mat_4, d=d)
        # imgfeature_tes_5 = add_contectual_feature(image_set=image_tes_mean, d=d)
        # X_tes_5 = numpy.concatenate((X_tes_4, context_tes_5.reshape((-1, 8)), imgfeature_tes_5.reshape((-1, 8))), axis=1)
        #
        # print('  predicting test set.', end='')
        # temp_score_tes_pred = clf_5.predict_proba(X_tes_5)[:, 1]
        # metric_auc = roc_auc_score(y_true=Y_tes, y_score=temp_score_tes_pred)
        # print('  AUC = {:.3f}'.format(metric_auc))
        #
        # print('  training set.', end='')
        # temp_score_tra_pred = clf_5.predict_proba(X_tra_5)[:, 1]
        # metric_auc = roc_auc_score(y_true=Y_tra, y_score=temp_score_tra_pred)
        # print('  AUC = {:.3f}\n'.format(metric_auc))
        #
        # print('pause')


        '''
        temp_ind_tes = 2  # demo
        temp_X_tes = image_dataset_tes[temp_ind_tes]['image'].reshape((-1, 3))
        image_size = image_dataset_tra[temp_ind_tes]['image_size']

        temp_score_tes_pred = clf.predict_proba(temp_X_tes)[:, 1]
        temp_Y_tes_pred = numpy.zeros_like(temp_score_tes_pred)
        temp_Y_tes_pred[temp_score_tes_pred >= 0.5] = 1

        temp_seg_pred = temp_Y_tes_pred.reshape(image_size)

        plt.figure(figsize=(3 * window_width, 1 * window_width))
        plt.subplot(1, 3, 1)
        plt.imshow(image_dataset_tes[temp_ind_tes]['image'][:, :, 0], cmap='gray')
        plt.subplot(1, 3, 2)
        plt.imshow(image_dataset_tes[temp_ind_tes]['seg'][:, :, 0], cmap='gray')
        plt.subplot(1, 3, 3)
        plt.imshow(temp_seg_pred, cmap='gray')
        plt.show()
        '''

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


if __name__ == '__main__':
    main()

