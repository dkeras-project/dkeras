#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import print_function, division

import os
import subprocess
import time

import ray

from dkeras.utils.sys_functions import get_port, get_addr


def _cmd(cmd):
    return subprocess.check_output(cmd.split(' ')).decode()


def _which_qsub():
    return subprocess.check_output(['which', 'qsub']).decode().replace('\n', '')


def _which_qstat():
    return subprocess.check_output(['which', 'qstat']).decode().replace('\n', '')


def _which_qdel():
    return subprocess.check_output(['which', 'qdel']).decode().replace('\n', '')


create_worker_script = """
ray start --redis-address={}:{}
sleep {}
"""


@ray.remote
def _get_n_nodes():
    return len(ray.nodes())


def wait_for_workers(n_workers, timeout=300):
    start_time = time.time()
    print("Waiting for {} workers".format(n_workers))
    while True:
        n_nodes = ray.get(_get_n_nodes.remote())
        if n_nodes >= n_workers:
            return True
        if (start_time - time.time() >= timeout):
            return False


def rm_existing_workers(qstat_path='qstat', qdel_path='qdel'):
    cmd = "{} | grep worker_script | cut - d ' ' -f1 | xargs {}".format(
        qstat_path, qdel_path)
    os.system(cmd)


def init_pbs_ray(n_workers=3, rm_existing=True, iface_name='eno1', worker_time=3600, verbose=True):
    """

    :param n_workers:
    :param rm_existing:
    :param iface_name:
    :param worker_time:
    :return:
    """
    if ray.is_initialized():
        if rm_existing:
            _cmd('ray stop')

    qsub_path = _which_qsub()
    qstat_path = _which_qstat()
    qdel_path = _which_qdel()

    # rm_existing_workers(qstat_path=qstat_path, qdel_path=qdel_path)
    rm_existing_workers()

    addresses = get_addr('eno1')
    addr = addresses[0]

    if verbose:
        print("Address: ", addr)
    if addr == 'No IP addr':
        raise Exception("Address not found for {}".format(iface_name))

    port = get_port()[1]
    if verbose:
        print("Port ", port)
        print("ray start --head --redis-port={}".format(port))
    _cmd('ray start --head --redis-port={}'.format(port))

    temp_dir = 'temp_{}'.format('_'.join(str(time.time()).split('.')))
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    worker_file = os.path.join(temp_dir, 'worker_script')
    worker_script = create_worker_script.format(addr, port, worker_time)
    with open(worker_file, 'w') as f:
        f.write(worker_script)

    if verbose:
        print("Worker file ", worker_file)

    qsub_pids = []
    for i in range(n_workers):
        if verbose:
            print("{} {}".format(qsub_path, worker_file))
            print(list(qsub_path))
            print(list(worker_file))

        qsub_pid = subprocess.check_output([qsub_path, '-lselect=1', '-lplace=excl', worker_file])
        qsub_pid = qsub_pid.decode()[:-1].split('.')[0]
        qsub_pids.append(qsub_pid)

    ray.init(redis_address='{}:{}'.format(addr, port))
    print("Ray initialized")
    return wait_for_workers(n_workers + 1), qsub_pids


def main():
    init_pbs_ray()
    print(ray.nodes())


if __name__ == "__main__":
    main()
