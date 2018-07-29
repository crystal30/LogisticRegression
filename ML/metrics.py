import numpy as np
from math import sqrt


def accuracy_score(y_true, y_predict):
    """计算y_true和y_predict之间的准确率"""
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"

    return np.sum(y_true == y_predict) / len(y_true)


def mean_squared_error(y,y_predict):
    assert y.shape == y_predict.shape, \
        "the size of y must be equal to the size of the y_predict"

    return sum((y - y_predict)**2)/len(y)

def root_mean_squared_error(y,y_predict):

    return sqrt(mean_squared_error(y,y_predict))

def mean_absolute_error(y,y_predict):
    assert y.shape == y_predict.shape, \
        "the size of y must be equal to the size of the y_predict"

    return sum(np.absolute(y - y_predict))/len(y)

def r2_score(y,y_predict):

    return 1-(mean_squared_error(y,y_predict)/np.var(y))




