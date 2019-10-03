#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import division, print_function

import os
import time

import numpy as np
import ray

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from dkeras.servers.data_server import DataServer
from dkeras.workers.worker import worker_task
from dkeras.config import config


class dKeras(object):
    """Distributed Keras Model Wrapper.

        It will automatically set up remote
        workers and data servers for data parallelism algorithms. Using
        the same notation as a regular Keras model, it makes distributing a
        Keras model simple.

        .. code-block:: python

            from tensorflow.keras.applications import ResNet50
            from dkeras import dKeras

            model = dKeras(ResNet50)
            preds = model.predict(data)

        Arguments:
            model: Un-initialized Keras model
            verbose: Verbose setting boolean variable. Default is False.
            weights: Weights arg for prebuilt models, example: ResNet50(
                weights='imagenet'). Default is None.
            n_workers: Integer number of worker processes. If left None,
                then it will automatically find the an estimate of the optimal
                number of workers. Default is None.
            init_ray: Boolean arg for whether to initialize Ray within
                the model initialization. Default is False.
            rm_existing_ray: Boolean arg for whether to remove any
                existing Ray clusters. Default is True.
            rm_local_model: Boolean arg for whether to remove the local
                copy of the Keras model for memory conservation. Default is
                False.
            wait_for_workers: Boolean arg for whether to wait for all of
                the worker processes to initialize and connect to the data
                server.
            redis_address: In the case of initializing Ray inside of
                model initialization, the redis address is required for
                connecting to existing Ray clusters.
            n_cpus_per_worker: The integer number of CPUs per worker
                processes. If left None, it will allocate automatically. The
                default is None.
            n_gpus_per_worker: The integer or float number of GPUs per
                worker processes. If left None, it will allocate
                automatically. The default is None.
            n_cpus_per_server: The integer number of CPUs per data
                server. If left None, it will allocate automatically. The
                default is None.
            """

    def __init__(self,
                 model,
                 verbose: bool = True,
                 weights: str = None,
                 n_workers: int = None,
                 init_ray: bool = True,
                 distributed: bool = True,
                 rm_existing_ray: bool = False,
                 rm_local_model: bool = True,
                 wait_for_workers: bool = False,
                 redis_address: str = None,
                 n_cpus_per_worker: int = None,
                 n_gpus_per_worker: int = None,
                 n_cpus_per_server: int = None):

        config.N_CPUS_PER_SERVER = n_cpus_per_server
        config.N_CPUS_PER_WORKER = n_cpus_per_worker
        config.N_CPUS_PER_SERVER = n_gpus_per_worker
        self.verbose = verbose
        if init_ray:
            if ray.is_initialized():
                if rm_existing_ray:
                    ray.shutdown()
                    ray.init()
                else:
                    if redis_address is None:
                        raise UserWarning(
                            "Ray already initialized, rm_existing_ray is "
                            "False, and redis_address is None")
                    else:
                        ray.init(redis_address=redis_address)
            else:
                ray.init()

        if n_workers is None:
            # self.n_workers = max(1, psutil.cpu_count() - 2)
            self.n_workers = config.DEFAULT_N_WORKERS
        else:
            self.n_workers = n_workers
        worker_ids = []
        for i in range(self.n_workers):
            worker_ids.append('worker_{}'.format(i))

        self.distributed = distributed
        self.worker_ids = worker_ids
        self.model = model(weights=weights)
        self.input_shape = self.model.input_shape
        ds = DataServer.remote(self.n_workers, self.worker_ids)
        weights = self.model.get_weights()
        weights = ray.put(weights)
        if rm_local_model:
            del self.model
        else:
            self.__dict__.update(self.model.__dict__)
            for k in dir(self.model):
                try:
                    if not k in dir(self):
                        self.__dict__[k] = getattr(self.model, k)
                except AttributeError:
                    pass

        def make_model():
            return model()

        for i in range(self.n_workers):
            worker_id = self.worker_ids[i]
            worker_task.remote(worker_id, weights, ds, make_model)
        self.data_server = ds

        if wait_for_workers:
            while True:
                if self.is_ready():
                    break
                else:
                    time.sleep(1e-3)

    def predict(self, x,
                distributed=True,
                int8_cvrt=False,
                batch_size=None,
                verbose=0,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False):
        """Generates output predictions for the input samples.
                Computation is done in batches.
                # Arguments
                    x: Input data. It could be:
                        - A Numpy array (or array-like), or a list of arrays
                          (in case the model has multiple inputs).
                        - A dict mapping input names to the corresponding
                          array/tensors, if the model has named inputs.
                        - A generator or `keras.utils.Sequence` returning
                          `(inputs, targets)` or `(inputs, targets, sample weights)`.
                        - None (default) if feeding from framework-native
                          tensors (e.g. TensorFlow data tensors).
                    batch_size: Integer or `None`.
                        Number of samples per gradient update.
                        If unspecified, `batch_size` will default to 32.
                        Do not specify the `batch_size` is your data is in the
                        form of symbolic tensors, generators, or
                        `keras.utils.Sequence` instances (since they generate batches).
                    verbose: Verbosity mode, 0 or 1.
                    steps: Total number of steps (batches of samples)
                        before declaring the prediction round finished.
                        Ignored with the default value of `None`.
                    callbacks: List of `keras.callbacks.Callback` instances.
                        List of callbacks to apply during prediction.
                        See [callbacks](/callbacks).
                    max_queue_size: Integer. Used for generator or `keras.utils.Sequence`
                        input only. Maximum size for the generator queue.
                        If unspecified, `max_queue_size` will default to 10.
                    workers: Integer. Used for generator or `keras.utils.Sequence` input
                        only. Maximum number of processes to spin up when using
                        process-based threading. If unspecified, `workers` will default
                        to 1. If 0, will execute the generator on the main thread.
                    use_multiprocessing: Boolean. Used for generator or
                        `keras.utils.Sequence` input only. If `True`, use process-based
                        threading. If unspecified, `use_multiprocessing` will default to
                        `False`. Note that because this implementation relies on
                        multiprocessing, you should not pass non-picklable arguments to
                        the generator as they can't be passed easily to children processes.
                # Returns
                    Numpy array(s) of predictions.
                # Raises
                    ValueError: In case of mismatch between the provided
                        input data and the model's expectations,
                        or in case a stateful model receives a number of samples
                        that is not a multiple of the batch size.
                """
        """
        Run inference on a data batch, returns predictions

        Arguments:
            data: numpy array of images
            distributed: True for distributed inference, false for serial
            close: boolean value for whether to stop workers
        return: Predictions
        """
        if distributed:
            if int8_cvrt:
                self.data_server.set_datatype.remote('int8')
                x = np.asarray(x)
                x = np.uint8(x * 255)
            n_data = len(x)
            if n_data % self.n_workers > 0:
                self.data_server.set_batch_size.remote(
                    int(n_data / self.n_workers) + 1)
            else:
                self.data_server.set_batch_size.remote(
                    int(n_data / self.n_workers))
            infer_config = [batch_size, verbose, steps, callbacks, max_queue_size,
                            workers, use_multiprocessing]
            self.data_server.push_data.remote(x, mode='infer', infer_config=infer_config)
            while not ray.get(self.data_server.is_complete.remote()):
                time.sleep(1e-4)
            return ray.get(self.data_server.pull_results.remote())
        else:
            return self.model.predict(x)

    def close(self, stop_ray=False):
        """
        Close the Ray workers for the model

        Arguments:
            stop_ray: Boolean value for whether to close Ray cluster

        return: None
        """
        self.data_server.close.remote()
        if stop_ray:
            ray.shutdown()
        time.sleep(5e-2)

    def is_ready(self):
        """
        Wait for workers to initialize

        :return: True
        """
        return ray.get(self.data_server.all_ready.remote())

    def compile(self, optimizer: str,
                loss=None,
                metrics=None,
                loss_weights=None,
                sample_weight_mode=None,
                weighted_metrics=None,
                target_tensors=None):
        """

        :param optimizer:
        :param loss:
        :param metrics:
        :param loss_weights:
        :param sample_weight_mode:
        :param weighted_metrics:
        :param target_tensors:
        :return:
        """
        if self.distributed:
            compile_data = ray.put([optimizer,
                                    loss,
                                    metrics,
                                    loss_weights,
                                    sample_weight_mode,
                                    weighted_metrics,
                                    target_tensors])
            self.data_server.push_compile.remote(compile_data)
        else:
            self.model.compile(optimizer,
                               loss=loss,
                               metrics=metrics,
                               loss_weights=loss_weights,
                               sample_weight_mode=sample_weight_mode)

    def fit(self,
            x=None,
            y=None,
            distributed=None,
            method='weight_averaging',
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.0,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False):
        if distributed == None:
            distributed = self.distributed
        if distributed:
            raise NotImplementedError("model.fit for distributed is not implemented yet,\
                dKeras will be updated soon")
            x = ray.put(x)
            y = ray.put(y)
        else:
            self.model.fit(x=x,
                           y=y,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=verbose,
                           callbacks=callbacks,
                           validation_split=validation_split,
                           validation_data=validation_data,
                           shuffle=shuffle,
                           class_weight=class_weight,
                           sample_weight=sample_weight,
                           initial_epoch=initial_epoch,
                           steps_per_epoch=steps_per_epoch,
                           validation_steps=validation_steps,
                           validation_freq=validation_freq,
                           max_queue_size=max_queue_size,
                           workers=workers,
                           use_multiprocessing=use_multiprocessing)

    def evaluate(self,
                 x=None,
                 y=None,
                 distributed=None,
                 batch_size=None,
                 verbose=1,
                 sample_weight=None,
                 steps=None,
                 callbacks=None,
                 max_queue_size=10,
                 workers=1,
                 use_multiprocessing=False):
        """Returns the loss value & metrics values for the model in test mode.
        Computation is done in batches.
        # Arguments
            x: Input data. It could be:
                - A Numpy array (or array-like), or a list of arrays
                  (in case the model has multiple inputs).
                - A dict mapping input names to the corresponding
                  array/tensors, if the model has named inputs.
                - A generator or `keras.utils.Sequence` returning
                  `(inputs, targets)` or `(inputs, targets, sample weights)`.
                - None (default) if feeding from framework-native
                  tensors (e.g. TensorFlow data tensors).
            y: Target data. Like the input data `x`,
                it could be either Numpy array(s), framework-native tensor(s),
                list of Numpy arrays (if the model has multiple outputs) or
                None (default) if feeding from framework-native tensors
                (e.g. TensorFlow data tensors).
                If output layers in the model are named, you can also pass a
                dictionary mapping output names to Numpy arrays.
                If `x` is a generator, or `keras.utils.Sequence` instance,
                `y` should not be specified (since targets will be obtained
                from `x`).
            batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` is your data is in the
                form of symbolic tensors, generators, or
                `keras.utils.Sequence` instances (since they generate batches).
            verbose: 0 or 1. Verbosity mode.
                0 = silent, 1 = progress bar.
            sample_weight: Optional Numpy array of weights for
                the test samples, used for weighting the loss function.
                You can either pass a flat (1D)
                Numpy array with the same length as the input samples
                (1:1 mapping between weights and samples),
                or in the case of temporal data,
                you can pass a 2D array with shape
                `(samples, sequence_length)`,
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify
                `sample_weight_mode="temporal"` in `compile()`.
            steps: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring the evaluation round finished.
                Ignored with the default value of `None`.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during evaluation.
                See [callbacks](/callbacks).
            max_queue_size: Integer. Used for generator or `keras.utils.Sequence`
                input only. Maximum size for the generator queue.
                If unspecified, `max_queue_size` will default to 10.
            workers: Integer. Used for generator or `keras.utils.Sequence` input
                only. Maximum number of processes to spin up when using
                process-based threading. If unspecified, `workers` will default
                to 1. If 0, will execute the generator on the main thread.
            use_multiprocessing: Boolean. Used for generator or
                `keras.utils.Sequence` input only. If `True`, use process-based
                threading. If unspecified, `use_multiprocessing` will default to
                `False`. Note that because this implementation relies on
                multiprocessing, you should not pass non-picklable arguments to
                the generator as they can't be passed easily to children processes.
        # Raises
            ValueError: in case of invalid arguments.
        # Returns
            Scalar test loss (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        """
        if distributed is None:
            distributed = self.distributed
        if distributed:
            raise NotImplementedError("model.evaulate is not yet supported with distributed=True")
        else:
            self.model.evaluate(x=x,
                                y=y,
                                batch_size=batch_size,
                                verbose=verbose,
                                sample_weight=sample_weight,
                                steps=steps,
                                callbacks=callbacks,
                                max_queue_size=max_queue_size,
                                workers=workers,
                                use_multiprocessing=use_multiprocessing)
