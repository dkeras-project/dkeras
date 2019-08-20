#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import division, print_function

import time

import numpy as np
from tensorflow.keras.applications import ResNet50

from dkeras import dKeras
import ray


def main():
    ray.init()
    n_data = 20
    model = dKeras(ResNet50, weights='imagenet', wait_for_workers=True, n_workers=3)

    for i in range(5):
        test_data = np.random.uniform(-1, 1, (n_data, 224, 224, 3))
        start_time = time.time()
        preds_1 = model.predict(test_data)
        elapsed = time.time() - start_time
        print(np.asarray(preds_1).shape)
        print("Time elapsed: {}\nFPS: {}".format(elapsed, n_data / elapsed))



    model.close()



if __name__ == "__main__":
    main()
