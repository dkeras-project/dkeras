#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import division, print_function

import argparse
import time
import ray

import numpy as np
from tensorflow.keras.applications import (DenseNet121, DenseNet169,
                                           DenseNet201,
                                           InceptionResNetV2, InceptionV3,
                                           MobileNet, MobileNetV2, NASNetLarge,
                                           NASNetMobile, ResNet50, VGG16, VGG19,
                                           Xception)

from dkeras import dKeras


def main():
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    model_names = {
        'densenet121'        : DenseNet121,
        'densenet169'        : DenseNet169,
        'densenet201'        : DenseNet201,
        'inception_v3'       : InceptionV3,
        'inception_resnet_v2': InceptionResNetV2,
        'mobilenet'          : MobileNet,
        'mobilenet_v2'       : MobileNetV2,
        'nasnet_large'       : NASNetLarge,
        'nasnet_mobile'      : NASNetMobile,
        'resnet50'           : ResNet50,
        'vgg16'              : VGG16,
        'vgg19'              : VGG19,
        'xception'           : Xception
    }
    # os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_data", help="Number of fake datapoints",
                        default=1000, type=int)
    parser.add_argument("--n_workers", help="Number of Ray workers",
                        default=5, type=int)

    parser.add_argument("--test", help="0: Local, 1: dKeras",
                        default=1, type=int)

    parser.add_argument("--search", help="True or False, Find best n_workers",
                        default=False, type=bool)

    parser.add_argument("--search-pool",
                        help="n workers to search from, separated by commas",
                        default='2,3,5,10,20,25,50,60,100', type=str)

    parser.add_argument("--model", help="Model options: {}".format(
        model_names.keys()),
                        default='resnet50', type=str)
    args = parser.parse_args()

    n_workers = args.n_workers
    test_type = args.test
    n_data = args.n_data
    model_name = args.model
    use_search = args.search
    search_pool = args.search_pool
    try:
        search_pool = list(map(int, search_pool.split(',')))
    except TypeError:
        raise UserWarning("Search pool arg must be int separated by commas")

    if not (model_name in model_names.keys()):
        raise UserWarning(
            "Model name not found: {}, options: {}".format(
                model_name, model_names))

    test_data = np.float16(np.random.uniform(-1, 1, (n_data, 224, 224, 3)))

    if test_type == 0:
        model = model_names[model_name]()

        start_time = time.time()
        preds = model.predict(test_data)
        elapsed = time.time() - start_time

        print("Time elapsed: {}\nFPS: {}".format(elapsed, n_data / elapsed))

    elif test_type == 1:
        if model_name != 'all':
            models = [model_names[model_name]]
        else:
            models = [model_names[m] for m in model_names.keys()]

        for m in models:
            if use_search:
                results = {}
                best_time = np.inf
                best_n_workers = -1

                for n in search_pool:
                    model = dKeras(m, wait_for_workers=True,
                                   init_ray=False,
                                   rm_existing_ray=False,
                                   n_workers=n)
                    print("Workers are ready")

                    start_time = time.time()
                    preds = model.predict(test_data)
                    elapsed = time.time() - start_time

                    time.sleep(3)

                    if elapsed < best_time:
                        best_time = elapsed
                        best_n_workers = n

                    results[str(n)] = elapsed
                    model.close()

                print('{}\nN\tElapsed Time'.format('=' * 80))

                for k in results.keys():
                    print("{}\t{}".format(k, results[k]))

                print("{}\nTests completed:\n\tBest N workers: {}\t FPS: {}".format(
                    '=' * 80, best_n_workers, n_data / best_time))
            else:
                model = dKeras(model_names[model_name],
                               init_ray=False,
                               wait_for_workers=True,
                               rm_existing_ray=False,
                               n_workers=n_workers)

            start_time = time.time()
            preds = model.predict(test_data)
            elapsed = time.time() - start_time

            model.close()
            time.sleep(3)

            print("Time elapsed: {}\nFPS: {}".format(elapsed, n_data / elapsed))


if __name__ == "__main__":
    main()
