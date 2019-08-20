#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import print_function, division

# -----------------------------------------------------------------------------
# Remote Worker Functions Configuration
# -----------------------------------------------------------------------------
DEFAULT_N_WORKERS = 4
WORKER_WAIT_TIME = 5e-3
WORKER_INFERENCE_BATCH_SIZE = 250

# -----------------------------------------------------------------------------
# Hardware Resource Configuration
# -----------------------------------------------------------------------------
N_CPUS_PER_SERVER = None
N_CPUS_PER_WORKER = 2
N_GPUS_PER_WORKER = 0

# -----------------------------------------------------------------------------
# Autoscaler Variables
# -----------------------------------------------------------------------------
CPU_OVERLOAD_LIMIT = 0.8