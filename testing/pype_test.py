#!/bin/env/python
#-*- encoding: utf-8 -*-
"""

"""
from __future__ import print_function, division
import pype
import ray

from tensorflow.keras.applications import ResNet50

def main():
    pype.init_ray()
    server = pype.Server.remote()
    server.add.remote('data', use_locking=False)
    server.add.remote('weights', use_locking=False)



if __name__ == '__main__':
    main()


