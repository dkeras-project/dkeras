#!/bin/env/python
#-*- encoding: utf-8 -*-
"""

"""
from __future__ import print_function, division
import argparse
import math
import time

import numpy as np
import ray
from keras.applications.resnet50 import ResNet50


def make_model():
    return ResNet50(weights='imagenet')


@ray.remote
def worker_task(ps, wait_time=1e-2):
    model = make_model()
    model_weights = ray.get(ps.fetch_model_weights.remote())
    model.set_weights(model_weights)
    while True:
        batch = ray.get(ps.fetch_batch.remote())
        if batch is None:
            time.sleep(wait_time)
        elif isinstance(batch, int):
            if batch == -1:
                return []
        else:
            return model.predict(batch)


@ray.remote
class FedInfParameterServer():

    def __init__(self, model_weights: np.ndarray, n_workers: int = 5):
        self.weights = model_weights
        self.data = None
        self.n_workers = n_workers

    def load_data(self, data: object) -> object:
        assert isinstance(data, np.ndarray), TypeError(
            "Invalid data arg type: {}".format(type(data).__name__))
        self.n_data = len(data)
        self.data = data

        # TODO: Better batch size method
        self.worker_batch_size = math.ceil(self.n_data / self.n_workers)

    def fetch_model_weights(self):
        return self.weights

    def fetch_batch(self):

        if self.data is None:
            return None
        if len(self.data) > 0:
            output = self.data[-self.worker_batch_size:]
            self.data = self.data[:-self.worker_batch_size]
            return output
        else:
            return -1


class FedInfer():

    def __init__(self, model, n_workers):
        self.ps = FedInfParameterServer.remote(model.get_weights(),
                                               n_workers=n_workers)

        self.workers = [worker_task.remote(self.ps) for _ in range(n_workers)]

    def infer(self, data):
        self.ps.load_data.remote(data)
        return ray.get(self.workers)


# def FedInfer(model, data, n_workers: int = 5):
#
#     ps = FedInfParameterServer.remote(model.get_weights(), n_workers=n_workers)
#
#     ps.load_data.remote(data)
#
#     results = ray.get([worker_task.remote(ps) for _ in range(n_workers)])
#
#     return results


def create_fake_data(size=1000):
    return np.random.uniform(0, 1, (size, 224, 224, 3))


def local_inference_test(n_data):
    data = create_fake_data(size=n_data)
    model = make_model()
    start_time = time.time()
    results = model.predict(data)
    time_elapsed = time.time() - start_time

    print("{}\nLocal inference test elapsed time: {}".format('='*80,time_elapsed))
    print("Local inference fps: {}\n{}".format(n_data / time_elapsed,"="80))


def fedinf_inference_test(n_workers, n_data):
    ray.init()

    data = create_fake_data(size=n_data)
    model = make_model()

    finfer = FedInfer(model, n_workers)

    print('-' * 80)

    start_time = time.time()
    results = finfer.infer(data)
    time_elapsed = time.time() - start_time

    print("{}\nFedInf inference test elapsed time: {}".format('='*80,time_elapsed))
    print("FedInf inference fps: {}\n{}".format(n_data / time_elapsed, '='*80))

    print(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_data", help="Number of fake datapoints",
                        default=1000, type=int)
    parser.add_argument("--n_workers", help="Number of Ray workers",
                        default=5, type=int)

    parser.add_argument("--test", help="0: Both, 1: Local, 2: Ray",
                        default=0, type=int)
    args = parser.parse_args()

    n_workers = args.n_workers
    test_type = args.test
    n_data = args.n_data

    if test_type == 0:
        local_inference_test(n_data)
        fedinf_inference_test(n_workers, n_data)
    elif test_type == 1:
        local_inference_test(n_data)
    elif test_type == 2:
        fedinf_inference_test(n_workers, n_data)
    else:
        raise UserWarning("Invalid test type arg: {}".format(test_type))


if __name__ == "__main__":
    main()

"""

================================================================================
Default
Local inference test elapsed time: 217.04220008850098
Local inference fps: 4.607398927914667
================================================================================

================================================================================
N workers = 5
FedInf inference test elapsed time: 45.0706250667572
FedInf inference fps: 22.187400297174296
================================================================================

================================================================================
N workers = 10
FedInf inference test elapsed time: 38.401707887649536
FedInf inference fps: 26.040508482738925
================================================================================

================================================================================
N workers = 20
FedInf inference test elapsed time: 29.663200855255127
FedInf inference fps: 33.71180355348739
================================================================================

================================================================================
N workers = 50
FedInf inference test elapsed time: 28.106022596359253
FedInf inference fps: 35.57956294141513
================================================================================

DevCloud with TF 1.14.0rc0
================================================================================
Local inference test elapsed time: 35.708417654037476
Local inference fps: 28.004601315256885
================================================================================

n_workers = 5
================================================================================
FedInf inference test elapsed time: 38.13867521286011
FedInf inference fps: 26.220103200197332
================================================================================

n_workers = 10
================================================================================
FedInf inference test elapsed time: 37.46253061294556
FedInf inference fps: 26.69333821390165
================================================================================
"""

