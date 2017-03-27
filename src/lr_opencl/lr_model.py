import numpy as np
import pandas as pd
import pyopencl as pcl



class LogisticRegression(object):
    """
    A model for fitting logistic regression using pyopencl for model fitting
    with a parallelized implementation of mini-batch gradient descent
    """

    def __init__(self,
                 batch_size=256,
                 regularization=None,
                 compute_device_preference='GPU'):

        self.batch_size = batch_size
        self.regularization = regularization
        self.compute_device_preference=compute_device_preference

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, bs):
        if not isinstance(bs, int):
            raise TypeError("'batch_size' must be an int")
        else:
            self._batch_size = bs

    @property
    def regularization(self):
        return self._regularization

    @regularization.setter
    def regularization(self, reg):
        avail_regs = ['l1', 'l2', None]
        if reg not in avail_regs:
            raise TypeError("'regularization' must be one of {x}".format(x=avail_regs))
        else:
            self._regularization = reg

    @property
    def compute_device_preference(self):
        return self._compute_device_preference

    @compute_device_preference.setter
    def compute_device_preference(self, cpd):
        if cpd is None:
            self._compute_device_preference = 'GPU'
        if isinstance(cpd, basestring): cpd = cpd.upper()
        pref_types = ['CPU', 'GPU']
        if cpd not in pref_types:
            raise TypeError("'compute_device_preference' must be either 'CPU' or 'GPU'")
        else:
            self.compute_device_preference = cpd


    def fit(self,X,y):
        # TODO: implement code here to fit model parameters on the GPU

        if isinstance(X, pd.DataFrame):
            X = X.values
        elif isinstance(X, pd.Series):
            X = X.to_frame().values
        elif not isinstance(X, np.ndarray):
            raise TypeError("'X' must be a pandas.DataFrame, pandas.Series or numpy.ndarray")

        # make sure that y is either one dimensional, or has only one row or
        # one column

        if isinstance(y, pd.DataFrame):
            y = X.values
        elif isinstance(y, pd.Series):
            y = X.to_frame().values
        elif not isinstance(y, np.ndarray):
            raise TypeError("'y' must be a pandas.DataFrame, pandas.Series or numpy.ndarray")

        if len(y.shape) == 2:
            is_one = [axislen == 1 for axislen in y.shape]
            if not any(is_one):
                raise TypeError("at least one dimension in 'y' must be equal to 1")
            else:
                y.shape = (y.size,)
        elif len(y.shape) > 2:
            raise TypeError("'y' must be either an order-1 array or an order-2 array with the size of one of the orders (rows or columns) equal to 1.")


        # start the fitting process


        return None


    def predict(X):
        # TODO: implement code here to run predictions on the GPU with OpenCL
        # or run on the CPU and python
        
        return None
