import numpy
import pandas
import os
import time


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

    demo_slide = {'fold': 3, 'n': 2}  # info of slide chosen to demonstrate

    n_sample = 10000
    n_tree = 20

    dataIO = DataPartition(path=data_path, val_ratio=0)

    if os.path.exists('./result/results.csv') is False:  # initialization
        df = pandas.DataFrame(columns=['fold 1', 'fold 2', 'fold 3', 'fold 4', 'fold 5'],
                              index=['baseline', '+superpixel', '+filtered'])
        df.to_csv('./result/results.csv')

    df = pandas.read_csv('./result/results.csv', index_col=0)
    print('existing csv file:')
    print(df)
    print('')

    results_bl = {'fold 1': 0, 'fold 2': 0, 'fold 3': 0, 'fold 4': 0, 'fold 5': 0}  # baseline
    results_bl_sp = {'fold 1': 0, 'fold 2': 0, 'fold 3': 0, 'fold 4': 0, 'fold 5': 0}  # + super pixel
    results_bl_sp_fi = {'fold 1': 0, 'fold 2': 0, 'fold 3': 0, 'fold 4': 0, 'fold 5': 0}  # + some low level image features

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

        image_tra_fea = add_image_feature(image_set=image_tra_mean)  # low level image filtering features
        image_tes_fea = add_image_feature(image_set=image_tes_mean)

        image_tra_pre, sp_tra = preprocess_image(image_set=image_dataset_tra.dataset['image'])  # super pixel
        image_tes_pre, sp_tes = preprocess_image(image_set=image_dataset_tes.dataset['image'])

        X_tra = image_dataset_tra.dataset['image'].reshape((-1, 3))
        X_tes = image_dataset_tes.dataset['image'].reshape((-1, 3))
        print('')


        # -----------------------------------------------------------------
        print('baseline')

        temp_all_idx = numpy.arange(0, Y_tra.shape[0])  # select only a subset of training set for training
        temp_idx_select_pos = numpy.random.choice(temp_all_idx[Y_tra > 0], size=n_sample, replace=True)
        temp_idx_select_neg = numpy.random.choice(temp_all_idx[Y_tra < 1], size=n_sample, replace=True)
        idx_select_0 = numpy.sort(numpy.concatenate((temp_idx_select_pos, temp_idx_select_neg)))

        clf_0 = RandomForestClassifier(n_estimators=n_tree, min_samples_leaf=200, max_depth=None, max_features='auto')
        clf_0.fit(X_tra[idx_select_0, :], Y_tra[idx_select_0])

        print('  predicting training set.', end='')
        score_tra_pred_0 = clf_0.predict_proba(X_tra)[:, 1]
        metric_auc = roc_auc_score(y_true=Y_tra, y_score=score_tra_pred_0)
        print('  AUC = {:.3f}'.format(metric_auc))

        print('  ================')
        print('  test set.', end='')
        temp_score_tes_pred = clf_0.predict_proba(X_tes)[:, 1]
        metric_auc = roc_auc_score(y_true=Y_tes, y_score=temp_score_tes_pred)
        print('  AUC = {:.3f}\n'.format(metric_auc))
        results_bl['fold {:01d}'.format(i_fold+1)] = metric_auc

        if i_fold == demo_slide['fold']:
            temp_ind_tes = demo_slide['n']  # demo
            demo_score_mat = temp_score_tes_pred.reshape((n_image_tes, ) + image_size)[temp_ind_tes, :, :]
            demo_seg_pred = numpy.zeros_like(demo_score_mat)
            demo_seg_pred[demo_score_mat >= 0.5] = 1
            demo_seg = image_dataset_tes[temp_ind_tes]['seg'][:, :, 0]
            demo_metric = roc_auc_score(y_true=demo_seg.flatten(), y_score=demo_score_mat.flatten())
            plt.figure(figsize=(3 * window_width, 1 * window_width))
            plt.subplot(1, 3, 1)
            plt.imshow(image_dataset_tes[temp_ind_tes]['image'][:, :, :])
            plt.title('image')
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(demo_seg, cmap='gray')
            plt.title('ground truth')
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(demo_seg_pred, cmap='gray')
            plt.title('prdiction AUC = {:.3f}'.format(demo_metric))
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('./result/demo_baseline.png')


        # -----------------------------------------------------------------
        print('+ super pixel features')
        X_tra_1 = numpy.concatenate((X_tra, image_tra_pre.reshape(-1, 3)), axis=1)

        temp_all_idx = numpy.arange(0, Y_tra.shape[0])  # select only a subset of training set for training
        temp_idx_select_pos = numpy.random.choice(temp_all_idx[Y_tra > 0], size=n_sample, replace=True)
        temp_idx_select_neg = numpy.random.choice(temp_all_idx[Y_tra < 1], size=n_sample, replace=True)
        idx_select_1 = numpy.sort(numpy.concatenate((temp_idx_select_pos, temp_idx_select_neg)))

        clf_1 = RandomForestClassifier(n_estimators=n_tree, min_samples_leaf=200, max_depth=None, max_features='auto')
        clf_1.fit(X_tra_1[idx_select_1, :], Y_tra[idx_select_1])

        X_tes_1 = numpy.concatenate((X_tes, image_tes_pre.reshape(-1, 3)), axis=1)

        print('  predicting training set.', end='')
        score_tra_pred_1 = clf_1.predict_proba(X_tra_1)[:, 1]
        metric_auc = roc_auc_score(y_true=Y_tra, y_score=score_tra_pred_1)
        print('  AUC = {:.3f}'.format(metric_auc))

        print('  ================')
        print('  test set.', end='')
        temp_score_tes_pred = clf_1.predict_proba(X_tes_1)[:, 1]
        metric_auc = roc_auc_score(y_true=Y_tes, y_score=temp_score_tes_pred)
        print('  AUC = {:.3f}\n'.format(metric_auc))
        results_bl_sp['fold {:01d}'.format(i_fold+1)] = metric_auc

        if i_fold == demo_slide['fold']:
            temp_ind_tes = demo_slide['n']  # demo
            demo_score_mat = temp_score_tes_pred.reshape((n_image_tes, ) + image_size)[temp_ind_tes, :, :]
            demo_seg_pred = numpy.zeros_like(demo_score_mat)
            demo_seg_pred[demo_score_mat >= 0.5] = 1
            demo_seg = image_dataset_tes[temp_ind_tes]['seg'][:, :, 0]
            demo_metric = roc_auc_score(y_true=demo_seg.flatten(), y_score=demo_score_mat.flatten())
            plt.figure(figsize=(3 * window_width, 1 * window_width))
            plt.subplot(1, 3, 1)
            plt.imshow(image_dataset_tes[temp_ind_tes]['image'][:, :, :])
            plt.title('image')
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(demo_seg, cmap='gray')
            plt.title('ground truth')
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(demo_seg_pred, cmap='gray')
            plt.title('prdiction AUC = {:.3f}'.format(demo_metric))
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('./result/demo_superpixel.png')


        # -----------------------------------------------------------------
        print('+ super pixel features + some filter features')
        X_tra_1 = numpy.concatenate((X_tra, image_tra_pre.reshape(-1, 3), image_tra_fea.reshape(-1, 3)), axis=1)

        temp_all_idx = numpy.arange(0, Y_tra.shape[0])  # select only a subset of training set for training
        temp_idx_select_pos = numpy.random.choice(temp_all_idx[Y_tra > 0], size=n_sample, replace=True)
        temp_idx_select_neg = numpy.random.choice(temp_all_idx[Y_tra < 1], size=n_sample, replace=True)
        idx_select_1 = numpy.sort(numpy.concatenate((temp_idx_select_pos, temp_idx_select_neg)))

        clf_1 = RandomForestClassifier(n_estimators=n_tree, min_samples_leaf=200, max_depth=None, max_features='auto')
        clf_1.fit(X_tra_1[idx_select_1, :], Y_tra[idx_select_1])

        X_tes_1 = numpy.concatenate((X_tes, image_tes_pre.reshape(-1, 3), image_tes_fea.reshape(-1, 3)), axis=1)

        print('  predicting training set.', end='')
        score_tra_pred_1 = clf_1.predict_proba(X_tra_1)[:, 1]
        metric_auc = roc_auc_score(y_true=Y_tra, y_score=score_tra_pred_1)
        print('  AUC = {:.3f}'.format(metric_auc))

        print('  ================')
        print('  test set.', end='')
        temp_score_tes_pred = clf_1.predict_proba(X_tes_1)[:, 1]
        metric_auc = roc_auc_score(y_true=Y_tes, y_score=temp_score_tes_pred)
        print('  AUC = {:.3f}\n'.format(metric_auc))
        results_bl_sp_fi['fold {:01d}'.format(i_fold+1)] = metric_auc

        if i_fold == demo_slide['fold']:
            temp_ind_tes = demo_slide['n']  # demo
            demo_score_mat = temp_score_tes_pred.reshape((n_image_tes, ) + image_size)[temp_ind_tes, :, :]
            demo_seg_pred = numpy.zeros_like(demo_score_mat)
            demo_seg_pred[demo_score_mat >= 0.5] = 1
            demo_seg = image_dataset_tes[temp_ind_tes]['seg'][:, :, 0]
            demo_metric = roc_auc_score(y_true=demo_seg.flatten(), y_score=demo_score_mat.flatten())
            plt.figure(figsize=(3 * window_width, 1 * window_width))
            plt.subplot(1, 3, 1)
            plt.imshow(image_dataset_tes[temp_ind_tes]['image'][:, :, :])
            plt.title('image')
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(demo_seg, cmap='gray')
            plt.title('ground truth')
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(demo_seg_pred, cmap='gray')
            plt.title('prdiction AUC = {:.3f}'.format(demo_metric))
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('./result/demo_filtered.png')


        '''
        temp_ind_tes = 2  # demo
        temp_X_tes = image_dataset_tes[temp_ind_tes]['image'].reshape((-1, 3))
        image_size = image_dataset_tes[temp_ind_tes]['image_size']

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
        plt.savefig('./result/demo_')
        plt.show()
        '''

    index_name = 'baseline'
    df_bl = pandas.DataFrame(results_bl, index=[index_name, ])
    if index_name in df.index:
        df.update(df_bl)
    else:
        df = df.append(df_bl)

    index_name = '+superpixel'
    df_bl_sp = pandas.DataFrame(results_bl_sp, index=[index_name, ])
    if index_name in df.index:
        df.update(df_bl_sp)
    else:
        df = df.append(df_bl_sp)

    index_name = '+filtered'
    df_bl_sp_fi = pandas.DataFrame(results_bl_sp_fi, index=[index_name, ])
    if index_name in df.index:
        df.update(df_bl_sp_fi)
    else:
        df = df.append(df_bl_sp_fi)

    print(df)

    df.to_csv('./result/results.csv')


if __name__ == '__main__':
    main()

