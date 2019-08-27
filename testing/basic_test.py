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

    n_data = 80
    test_data = np.random.uniform(-1, 1, (n_data, 224, 224, 3))
    model = dKeras(ResNet50, weights='imagenet', wait_for_workers=True, n_workers=4)

    start_time = time.time()
    preds = model.predict(test_data)
    elapsed = time.time() - start_time

    model.close()

    time.sleep(3)

    model = ResNet50(weights='imagenet')
    preds2 = model.predict(test_data)

    # for i in range(n_data):
    #     for j,n in enumerate(preds[i]):
    #         print(n, preds2[i][j])
    #     print('-'*80)

    print("Time elapsed: {}\nFPS: {}".format(elapsed, n_data / elapsed))


if __name__ == "__main__":
    main()
