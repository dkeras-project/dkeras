#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import print_function, division

import multiprocessing as mp
import time

import matplotlib.pyplot as plt
import numpy as np
import psutil
from tensorflow.keras.applications import ResNet50


def run_predict():
    test_data = np.random.uniform(-1, 1, (1000, 224, 224, 3))
    model = ResNet50()
    time.sleep(1)
    model.predict(test_data)


def monitor(target):
    worker_process = mp.Process(target=target)
    worker_process.start()
    #p = psutil.Process(worker_process.pid)
    time.sleep(1)
    # log cpu usage of `worker_process` every 10 ms
    cpu_percents = []
    while worker_process.is_alive():
        cpu_percents.append(psutil.cpu_percent(percpu=False))
        time.sleep(1e-3)

    worker_process.join()
    return cpu_percents


def main():
    cpu_percents = monitor(target=run_predict)

    # Data for plotting
    t = np.asarray(range(len(cpu_percents)))
    cpu_percents = np.asarray(cpu_percents)

    for i in cpu_percents:
        print(i)

    # fig, ax = plt.subplots()
    # ax.plot(t, cpu_percents)
    #
    # ax.set(xlabel='time (ms)', ylabel='CPU utilization',
    #        title='CPU utilization for ResNet50 Inference')
    # ax.grid()
    #
    # fig.savefig("test.png")


if __name__ == "__main__":
    main()
