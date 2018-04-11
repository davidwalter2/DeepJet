import numpy as np

def ks_w(data1, data2, wei1, wei2):
    """
    from https://stackoverflow.com/questions/40044375/how-to-calculate-the-kolmogorov-smirnov-statistic-between-two-weighted-samples
    Kolmogorow-Smirnow-Test for weighted samples
    :param data1:
    :param data2:
    :param wei1:
    :param wei2:
    :return: Kolmogorow-Smirnow-Teststatistic
    """
    ix1 = np.argsort(data1)
    ix2 = np.argsort(data2)
    data1 = data1[ix1]
    data2 = data2[ix2]
    wei1 = wei1[ix1]
    wei2 = wei2[ix2]
    data = np.concatenate([data1, data2])
    cwei1 = np.hstack([0, np.cumsum(wei1)/sum(wei1)])
    cwei2 = np.hstack([0, np.cumsum(wei2)/sum(wei2)])
    cdf1we = cwei1[[np.searchsorted(data1, data, side='right')]]
    cdf2we = cwei2[[np.searchsorted(data2, data, side='right')]]
    return np.max(np.abs(cdf1we - cdf2we))

def shuffle_in_unison(listofArrays):
    '''
    :param list of arrays: a list of numpy arrays with the same length in the first dimension
    :return: list of arrays shuffled only in the first dimension
    '''
    shuffled_list = []
    for arr in listofArrays:
        shuffled_list.append(np.empty(arr.shape, dtype=arr.dtype))
    permutation = np.random.permutation(len(listofArrays[0]))
    for old_index, new_index in enumerate(permutation):
        for i, arr in enumerate(listofArrays):
            shuffled_list[i][new_index] = arr[old_index]
    return shuffled_list

def get_class_weights(y, smooth_factor=0):
    """
    Returns the weights for each class based on the frequencies of the samples
    :param smooth_factor: factor that smooths extremely uneven weights
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """
    counter = np.sum(y,axis=0,dtype='int32')

    if smooth_factor > 0:
        p = max(counter) * smooth_factor
        for k in range(len(counter)):
            counter[k] += p

    majority = max(counter)

    return {cls: float(majority) / count for cls, count in enumerate(counter)}


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, variance)