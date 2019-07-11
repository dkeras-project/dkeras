#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import print_function, division
import numpy as np
import time
import ray

from tensorflow.keras.applications import ResNet50


def local_inference_test(n_data):
    """

    :param n_data:
    :return:
    """
    data = np.random.uniform(-1, 1, (n_data, 224, 224, 3))
    model = ResNet50()
    start_time = time.time()
    model.predict(data)
    elapsed_time = time.time()-start_time
    del model
    print("{}\nLocal Inference Test")
    print("Frames: {}\nFPS: {}".format(n_data, n_data/elapsed_time))


def distributed_test(n_workers, n_data):
    """

    :param n_workers:
    :param n_data:
    :return:
    """
    ray.init()
    n_data = 1000
    data = np.random.uniform(-1, 1, (n_data, 224, 224, 3))
    model = dKeras(ResNet50, n_workers=n_workers)
    time.sleep(20)
    start_time = time.time()
    model.predict(data)
    elapsed_time = time.time()-start_time
    print("{}\nLocal Inference Test")
    print("Frames: {}\nFPS: {}".format(n_data, n_data / elapsed_time))


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
        distributed_test(n_workers, n_data)
    elif test_type == 1:
        local_inference_test(n_data)
    elif test_type == 2:
        distributed_test(n_workers, n_data)
    else:
        raise UserWarning("Invalid test type arg: {}".format(test_type))


if __name__ == "__main__":
    main()
