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
    model = ResNet50()
    test_data = np.random.uniform(-1, 1, (100, 224, 224, 3))
    time.sleep(1)
    model.predict(test_data)


def monitor(target):
    worker_process = mp.Process(target=target)
    worker_process.start()
    #p = psutil.Process(worker_process.pid)

    # log cpu usage of `worker_process` every 10 ms
    cpu_percents = []
    while worker_process.is_alive():
        cpu_percents.append(psutil.cpu_percent(percpu=False))
        time.sleep(1e-3)

    worker_process.join()
    return cpu_percents


def main():
    cpu_percents = monitor(target=run_predict)
    print(cpu_percents)

    # Data for plotting
    t = np.asarray(range(len(cpu_percents)))
    cpu_percents = np.asarray(cpu_percents)

    fig, ax = plt.subplots()
    ax.plot(t, cpu_percents)

    ax.set(xlabel='time (ms)', ylabel='CPU utilization',
           title='CPU utilization for ResNet50 Inference')
    ax.grid()

    fig.savefig("test.png")
    plt.show()


if __name__ == "__main__":
    main()
