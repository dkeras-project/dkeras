#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import division, print_function

import numpy as np
from tensorflow.keras.applications import ResNet50, MobileNet

from dkeras import dKeras
import ray


def main():
    """
    Test out using multiple models at the same time, this case is running
    ResNet50 and MobileNet
    """
    ray.init()
    n_data = 20

    model1 = dKeras(ResNet50, weights='imagenet', wait_for_workers=True, n_workers=3)
    model2 = dKeras(MobileNet, weights='imagenet', wait_for_workers=True, n_workers=3)

    test_data = np.random.uniform(-1, 1, (n_data, 224, 224, 3))

    model1.predict(test_data)
    model2.predict(test_data)

    model1.close()
    model2.close()

if __name__ == "__main__":
    main()
