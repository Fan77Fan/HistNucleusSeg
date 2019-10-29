import numpy
import scipy

from scipy.signal import convolve2d

import matplotlib.pyplot as plt


def add_image_feature(image_set):
    assert len(image_set.shape) == 3  # dimension: [n_image, X, Y]
    n_image = image_set.shape[0]
    # n_chan = prob_set.shape[1]
    image_size = image_set.shape[1:]

    new_image_set_list = []
    for i in range(n_image):
        image_list = []
        m = image_set[i, :, :]
        # image_list.append(convolve2d(m, filter(1), mode='same', boundary='wrap'))
        # image_list.append(convolve2d(m, filter(2), mode='same', boundary='wrap'))
        # image_list.append(convolve2d(m, filter(3), mode='same', boundary='wrap'))
        # image_list.append(convolve2d(m, filter(4), mode='same', boundary='wrap'))
        # image_list.append(convolve2d(m, filter(6), mode='same', boundary='wrap'))
        # image_list.append(convolve2d(m, filter(7), mode='same', boundary='wrap'))
        # image_list.append(convolve2d(m, filter(8), mode='same', boundary='wrap'))
        # image_list.append(convolve2d(m, filter(9), mode='same', boundary='wrap'))
        image_list.append(convolve2d(m, filter('dx'), mode='same', boundary='wrap'))
        image_list.append(convolve2d(m, filter('dy'), mode='same', boundary='wrap'))
        # image_list.append(convolve2d(m, filter('Ldx'), mode='same', boundary='wrap'))
        # image_list.append(convolve2d(m, filter('Ldy'), mode='same', boundary='wrap'))
        image_list.append(convolve2d(m, filter('laplacian'), mode='same', boundary='wrap'))
        new_image_set_list.append(numpy.stack(image_list, axis=2))

    image_set_aug = numpy.stack(new_image_set_list, axis=0)
    return image_set_aug


def add_contectual_feature(image_set):
    assert len(image_set.shape) == 3  # dimension: [n_image, X, Y]
    n_image = image_set.shape[0]
    # n_chan = prob_set.shape[1]
    image_size = image_set.shape[1:]

    context_list = []
    for i in range(n_image):
        temp_context_list = []
        m = image_set[i, :, :]
        temp_context_list.append(convolve2d(m, filter(1), mode='same', boundary='wrap'))
        temp_context_list.append(convolve2d(m, filter(2), mode='same', boundary='wrap'))
        temp_context_list.append(convolve2d(m, filter(3), mode='same', boundary='wrap'))
        temp_context_list.append(convolve2d(m, filter(4), mode='same', boundary='wrap'))
        temp_context_list.append(convolve2d(m, filter(6), mode='same', boundary='wrap'))
        temp_context_list.append(convolve2d(m, filter(7), mode='same', boundary='wrap'))
        temp_context_list.append(convolve2d(m, filter(8), mode='same', boundary='wrap'))
        temp_context_list.append(convolve2d(m, filter(9), mode='same', boundary='wrap'))
        # image_list.append(convolve2d(m, filter('dx'), mode='same', boundary='wrap'))
        # image_list.append(convolve2d(m, filter('dy'), mode='same', boundary='wrap'))
        # image_list.append(convolve2d(m, filter('Ldx'), mode='same', boundary='wrap'))
        # image_list.append(convolve2d(m, filter('Ldy'), mode='same', boundary='wrap'))
        # image_list.append(convolve2d(m, filter('laplacian'), mode='same', boundary='wrap'))
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