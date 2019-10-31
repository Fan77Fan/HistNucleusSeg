import numpy
import scipy

from scipy.signal import convolve2d
from skimage.segmentation import mark_boundaries, slic, felzenszwalb

import matplotlib.pyplot as plt


def preprocess_image(image_set):
    '''
    use super pixel image to process the raw image and compute features
    :param image_set: the array of images in dimension of [n_image, X, Y, channel]
    :return: numpy array of features, array of super pixel images
    '''
    n_image = image_set.shape[0]
    if len(image_set.shape) == 4:  # dimension: [n_image, X, Y, channel]
        image_size = image_set.shape[1:3]
        flag_chan = True
    elif len(image_set.shape) == 3:  # dimension: [n_image, X, Y]
        image_size = image_set.shape[1:]
        flag_chan = False
    else:
        raise Exception('dimension of image_set unsupported.')
    new_image_set = []
    sp_set = []
    for i in range(n_image):
        if flag_chan is True:
            img = image_set[i, :, :, :]
            m = img.mean(2) / 255
        else:
            img = image_set[i, :, :]
            m = img / 255
        sp = felzenszwalb(img, scale=1, sigma=0, min_size=100, multichannel=flag_chan)  # super pixel
        n_label = sp.max()
        m_ave = numpy.zeros(image_size)
        m_std = numpy.zeros(image_size)
        m_sze = numpy.zeros(image_size)
        for l in range(n_label):
            m_ave[sp == l] = m[sp == l].mean()
            m_std[sp == l] = m[sp == l].std()
            m_sze[sp == l] = (sp ==l).sum()
        new_image_set.append(numpy.stack((m_ave, m_std, m_sze), axis=2))
        sp_set.append(sp)
    new_image_set = numpy.stack(new_image_set, axis=0)
    sp_set = numpy.stack(sp_set, axis=0)
    return new_image_set, sp_set


def process_score_sp(image_set, sp_set):
    '''
    use super pixel to process the predicted score from a classifier
    and use that generate new features
    :param image_set: array of probability score in the dimension of [n_image, X, Y]
    :param sp_set: array of super pixel images generated from function preprocess_image
    :return: array of new features
    '''
    assert len(image_set.shape) == 3  # dimension: [n_image, X, Y]
    n_image = image_set.shape[0]
    image_size = image_set.shape[1:]
    new_image_set = []
    for i in range(n_image):
        m = image_set[i, :, :]
        sp =  sp_set[i, :, :]  # super pixel
        n_label = sp.max()
        m_ave = numpy.zeros(image_size)
        m_std = numpy.zeros(image_size)
        # m_sze = numpy.zeros(image_size)
        for l in range(n_label):
            m_ave[sp == l] = m[sp == l].mean()
            m_std[sp == l] = m[sp == l].std()
            # m_sze[sp == l] = (sp ==l).sum()
        new_image_set.append(numpy.stack((m_ave, m_std), axis=2))
    new_image_set = numpy.stack(new_image_set, axis=0)
    return new_image_set


def add_image_feature(image_set):
    '''
    generate low level image features using some simple filterings
    :param image_set: array of gray-scale images in the dimension of [n_image, X, Y]
    :return: array of new features
    '''
    assert len(image_set.shape) == 3  # dimension: [n_image, X, Y]
    n_image = image_set.shape[0]
    # n_chan = prob_set.shape[1]
    image_size = image_set.shape[1:]

    new_image_set_list = []
    for i in range(n_image):
        image_list = []
        m = image_set[i, :, :]
        image_list.append(convolve2d(m, filter('dx'), mode='same', boundary='wrap'))
        image_list.append(convolve2d(m, filter('dy'), mode='same', boundary='wrap'))
        # image_list.append(convolve2d(m, filter('Ldx'), mode='same', boundary='wrap'))
        # image_list.append(convolve2d(m, filter('Ldy'), mode='same', boundary='wrap'))
        image_list.append(convolve2d(m, filter('laplacian'), mode='same', boundary='wrap'))
        new_image_set_list.append(numpy.stack(image_list, axis=2))

    image_set_aug = numpy.stack(new_image_set_list, axis=0)
    return image_set_aug


def add_contectual_feature(image_set, d):
    '''
    generate features by sampling around the neighourhood pixels in the given array of images
    :param image_set: array of sampled images [n_image, X, Y]
    :param d: node to node distance in a 3x3 matrix
    :return: array of new features
    '''
    assert len(image_set.shape) == 3  # dimension: [n_image, X, Y]
    n_image = image_set.shape[0]
    # n_chan = prob_set.shape[1]
    image_size = image_set.shape[1:]

    context_list = []
    for i in range(n_image):
        temp_context_list = []
        m = image_set[i, :, :]
        for x in range(0, 3):  # add eight neighbours
            for y in range(0, 3):
                if x != 1 or y != 1:
                    h = numpy.zeros(shape=(2 * d + 1, 2 * d + 1))
                    h[x*d, y*d] = 1
                    temp_context_list.append(convolve2d(m, h, mode='same', boundary='wrap'))
        context_list.append(numpy.stack(temp_context_list, axis=2))

    context_mat = numpy.stack(context_list, axis=0)
    return context_mat


def filter(a):
    if a == 1:
        h = numpy.zeros(shape=(5, 5))
        h[0, 0] = 1
    elif a == 2:
        h = numpy.zeros(shape=(5, 5))
        h[0, 2] = 1
    elif a == 3:
        h = numpy.zeros(shape=(5, 5))
        h[0, 4] = 1
    elif a == 4:
        h = numpy.zeros(shape=(5, 5))
        h[2, 0] = 1
    elif a == 5:
        h = numpy.zeros(shape=(5, 5))
        h[2, 2] = 1
    elif a == 6:
        h = numpy.zeros(shape=(5, 5))
        h[2, 4] = 1
    elif a == 7:
        h = numpy.zeros(shape=(5, 5))
        h[4, 0] = 1
    elif a == 8:
        h = numpy.zeros(shape=(5, 5))
        h[4, 2] = 1
    elif a == 9:
        h = numpy.zeros(shape=(5, 5))
        h[4, 4] = 1
    elif a is 'dx':
        h = numpy.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    elif a is 'dy':
        h = numpy.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
    elif a is 'Ldx':
        h = numpy.zeros(shape=(5, 5))
        h[2, 0] = -1
        h[2, 4] = 1
    elif a is 'Ldy':
        h = numpy.zeros(shape=(5, 5))
        h[0, 2] = -1
        h[4, 2] = 1
    elif a is 'laplacian':
        g = numpy.zeros(shape=(5, 5))
        g[2, 2] = 1
        h = scipy.ndimage.filters.laplace(g)
    else:
        raise Exception('h argument not supported.')

    return h


