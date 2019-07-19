#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import print_function, division
from dkeras.config import config
import psutil


class HWProfiler(object):

    def __init__(self):
        pass

    @staticmethod
    def n_workers(n_servers: int) -> int:
        """

        :param n_servers:
        :return:
        """
        n_cpus = psutil.cpu_count()
        used = 1 + n_servers * config.N_CPUS_PER_SERVER
        if used > (n_cpus - config.N_CPUS_PER_WORKER):
            raise UserWarning(
                "CPU configurations too high for {} CPUs".format(n_cpus))
        return int((n_cpus - used) / config.N_CPUS_PER_WORKER)

    @staticmethod
    def is_overloaded():
        """

        :return:
        """
        return psutil.cpu_percent() > config.CPU_OVERLOAD_LIMIT
