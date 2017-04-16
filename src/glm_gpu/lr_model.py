import numpy as np
import pandas as pd
import pyopencl as pcl
from . import settings
from .cl_manager import OpenCLManager
import math

manager = OpenCLManager(settings.CL_FILE,
                        settings.CL_PREFERRED_DEVICE)

def _check_X(X):

    if isinstance(X, pd.DataFrame):
        X = X.values
    elif isinstance(X, pd.Series):
        X = X.to_frame().values
    elif not isinstance(X, np.ndarray):
        raise TypeError("'X' must be a pandas.DataFrame, pandas.Series or numpy.ndarray")

    return X

def _check_y(y):
    # make sure that y is either one dimensional, or has only one row or
    # one column
    #
    if len(y.shape) == 2:
        is_one = [axislen == 1 for axislen in y.shape]
        if not any(is_one):
            raise TypeError("at least one dimension in 'y' must be equal to 1")
        else:
            y = y.reshape((y.size, ))
    elif len(y.shape) > 2:
        raise TypeError("'y' must be either an order-1 array or an order-2 array with the size of one of the orders (rows or columns) equal to 1.")

    if isinstance(y, pd.DataFrame):
        y = y.values
    elif isinstance(y, pd.Series):
        y = y.to_frame().values
    elif not isinstance(y, np.ndarray):
        raise TypeError("'y' must be a pandas.DataFrame, pandas.Series or numpy.ndarray")

    return y





class LogisticRegressionGPU(object):
    """
    A model for fitting logistic regression using pyopencl for model fitting
    with a parallelized implementation of stochastic gradient descent
    """

    def __init__(self,
                 hdf_store,
                 table_key,
                 X_columns,
                 y_column,
                 batch_size=None,
                 regularization=None):

        if batch_size is None:
            self.batch_size = self._compute_best_batch_size(len(X_columns) + 1, manager.device)
        elif isinstance(batch_size, int):
            self.batch_size = batch_size

        self.store = hdf_store
        self.xcols = X_columns
        self.ycol = y_column
        self.all_cols = self.xcols.append(self.ycol)
        self.table = table_key
        self.regularization = regularization
        self.nrows = hdf_store.get_storer(table_key).nrows
        self.epoch_increment = float(self.batch_size) / float(self.nrows)
        self.cost=None



    def _compute_best_batch_size(self, total_columns, device, memory_load_percentage=0.3):
        # each cell is a 32 bit floating point number
        # so the memory needed for one row is 4 * total_columns (in bytes)
        # we want to use the numer of rows that gets us closest to the
        # requested memory load percentage that is also a multiple of the
        # devices max work group size

        nbytes_per_row = total_columns * 4
        optimal_mem_load = device.global_mem_size * memory_load_percentage
        max_wg_size = device.max_work_group_size

        best_batch_size = math.floor( optimal_mem_load / nbytes_per_row )
        rem = best_batch_size % max_wg_size
        best_batch_size -= rem
        best_batch_size = int(best_batch_size)

        return best_batch_size

    # TODO: make properties for table_key, store, x and y columns
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


    def fit(self,
            learning_rate=1e-2,
            train_size=0.6,
            n_epoch=2.0,
            cost_iteration=5):

        train_start_index = 0
        train_stop_index = int(math.floor(train_size * self.nrows))
        train_batch_size = self.batch_size
        loop_count = 0
        epoch_count = 0.0
        cost = []
        cur_train_index_start= train_start_index
        cur_train_index_stop = cur_train_index_start + train_batch_size
        self.theta = np.random.normal(size=(len(self.xcols,)))

        while epoch_count < n_epoch:

            subtable = self.store.select(self.table,
                                         start=cur_train_index_start,
                                         stop=cur_train_index_stop,
                                         columns=self.all_cols)

            X = subtable.loc[:, self.xcols]
            y = subtable.loc[:, self.ycol]

            if loop_count % cost_iteration == 0:
                c = self._gpu_cost(X, y, self.theta)
                cost.append(c)

            # compute the gradient
            gradient = self._gpu_compute_gradient(X, y, self.theta)
            # update
            self.theta += learning_rate * gradient
            # wow, such machine learning!
            epoch_count += self.epoch_increment
            loop_count += 1

        self.cost = cost
        return None

    def predict(X):
        X = _check_X(X)
        predictions = self._gpu_predict(X, self.theta)
        return predictions


    def _gpu_predict(self, X, theta):
        X = X.values.astype(np.float32)
        theta = theta.astype(np.float32)

        out = np.zeros(shape=(X.shape[0],), dtype=np.float32)

        X_buf = pcl.Buffer(manager.context, pcl.mem_flags.READ_ONLY | pcl.mem_flags.COPY_HOST_PTR, hostbuf=X)
        theta_buf = pcl.Buffer(manager.context, pcl.mem_flags.READ_ONLY | pcl.mem_flags.COPY_HOST_PTR, hostbuf=theta)
        out_buf = pcl.Buffer(manager.context, pcl.mem_flags.READ_WRITE | pcl.mem_flags.COPY_HOST_PTR, hostbuf=out)

        prediction_kernel = manager.kernel_dictionary['logisitc_prediction']

        nrows = np.int32(X.shape[0])
        ncols = np.int32(X.shape[1])

        prediction_kernel(manager.queue,
                          X.shape,
                          None,
                          X_buf,
                          theta_buf,
                          out_buf,
                          nrows,
                          ncols)

        prediction_kernel.wait()
        pcl.enqueue_copy(manager.queue, out, out_buf)
        return out


    def _gpu_compute_gradient(self, X, y, theta):
        X = X.values.astype(np.float32)
        y = y.values.astype(np.float32)
        theta = theta.astype(np.float32)

        global_reduction_groups = 64 * manager.device.max_work_group_size # can change this later
        gradient = np.zeros(shape=X.shape, dtype=np.float32)
        temp_gradient_average = np.zeros(shape=(global_reduction_groups, X.shape[1]), dtype=np.float32)
        model_estimate = np.zeros(shape=(X.shape[0],), dtype=np.float32)

        gradient_kernel = manager.kernel_dictionary['logistic_gradient_ols']
        row_avg_kernel = manager.kernel_dictionary['matrix_row_mean']

        X_buf = pcl.Buffer(manager.context, pcl.mem_flags.READ_ONLY | pcl.mem_flags.COPY_HOST_PTR, hostbuf=X)
        y_buf = pcl.Buffer(manager.context, pcl.mem_flags.READ_ONLY | pcl.mem_flags.COPY_HOST_PTR, hostbuf=y)
        theta_buf = pcl.Buffer(manager.context, pcl.mem_flags.READ_ONLY | pcl.mem_flags.COPY_HOST_PTR, hostbuf=theta)
        gradient_buf = pcl.Buffer(manager.context, pcl.mem_flags.READ_WRITE | pcl.mem_flags.COPY_HOST_PTR, hostbuf=gradient)
        model_estimate_buf = pcl.Buffer(manager.context, pcl.mem_flags.READ_WRITE | pcl.mem_flags.COPY_HOST_PTR, hostbuf=model_estimate)
        temp_gradient_average_buf = pcl.Buffer(manager.context, pcl.mem_flags.WRITE_ONLY, size=temp_gradient_average.nbytes)
        scratch_buf = pcl.LocalMemory(np.float32().nbytes * manager.device.max_work_group_size)

        nrows = np.int32(X.shape[0])
        ncols = np.int32(X.shape[1])

        gradient_event = gradient_kernel(manager.queue,
                                         X.shape,
                                         None,
                                         X_buf,
                                         theta_buf,
                                         y_buf,
                                         gradient_buf,
                                         nrows,
                                         ncols)

        avg_event = row_avg_kernel(manager.queue,
                                   (global_reduction_groups,),
                                   (manager.device.max_work_group_size,),
                                   gradient_buf,
                                   temp_gradient_average_buf,
                                   scratch_buf,
                                   nrows,
                                   ncols)

        gradient_event.wait()
        avg_event.wait()
        pcl.enqueue_copy(manager.queue, temp_gradient_average, temp_gradient_average_buf)
        gradient_avg = temp_gradient_average.mean(axis=0)
        return gradient_avg


    def _gpu_cost(self, X, y, theta):
        # computes the average cost
        X = X.values.astype(np.float32)
        y = y.values.astype(np.float32)
        theta = theta.astype(np.float32)

        cost_kernel = manager.kernel_dictionary['logistic_cost_ols']
        row_avg_kernel = manager.kernel_dictionary['matrix_row_mean']

        global_reduction_groups = 64 * manager.device.max_work_group_size # can change this later
        temp_average = np.zeros(shape=(global_reduction_groups,), dtype=np.float32)
        cost = np.zeros(shape=y.shape)

        X_buf = pcl.Buffer(manager.context, pcl.mem_flags.READ_ONLY | pcl.mem_flags.COPY_HOST_PTR, hostbuf=X)
        y_buf = pcl.Buffer(manager.context, pcl.mem_flags.READ_ONLY | pcl.mem_flags.COPY_HOST_PTR, hostbuf=y)
        theta_buf = pcl.Buffer(manager.context, pcl.mem_flags.READ_ONLY | pcl.mem_flags.COPY_HOST_PTR, hostbuf=theta)
        cost_buf = pcl.Buffer(manager.context, pcl.mem_flags.READ_WRITE | pcl.mem_flags.COPY_HOST_PTR, hostbuf=cost)
        temp_buf = pcl.Buffer(manager.context, pcl.mem_flags.WRITE_ONLY, size=temp_average.nbytes)
        scratch_buf = pcl.LocalMemory(np.float32().nbytes * manager.device.max_work_group_size)

        nrows = np.int32(X.shape[0])
        ncols = np.int32(X.shape[1])

        cost_event = cost_kernel(manager.queue,
                                 X.shape,
                                 None,
                                 X_buf,
                                 theta_buf,
                                 y_buf,
                                 cost_buf,
                                 nrows,
                                 ncols)

        avg_event = row_avg_kernel(manager.queue,
                                  (global_reduction_groups,),
                                  (manager.device.max_work_group_size,),
                                  cost_buf,
                                  temp_buf,
                                  scratch_buf,
                                  nrows,
                                  np.int32(1))

        cost_event.wait()
        avg_event.wait()
        pcl.enqueue_copy(manager.queue, temp_average, temp_buf).wait()
        cost_avg = temp_average.mean(axis=0)
        return cost_avg
