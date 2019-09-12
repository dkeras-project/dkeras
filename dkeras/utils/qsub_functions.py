#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import print_function, division
from dkeras.utils.sys_functions import get_port, get_addr
import subprocess
import time
import ray
import os


def _which_qsub():
    return subprocess.check_output(['which', 'qsub']).decode().replace('\n', ' ')


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
        if len(n_nodes) >= n_workers:
            return True
        if (start_time-time.time() >= timeout):
            return False


def init_pbs_ray(n_workers=3, rm_existing=True, iface_name='eno1', worker_time=3600):
    if ray.is_initialized():
        if rm_existing:
            ray.shutdown()
    qsub_path = _which_qsub()
    addresses = get_addr('eno1')
    addr = addresses[0]

    print("Address: ", addr)
    if addr == 'No IP addr':
        raise Exception("Address not found for {}".format(iface_name))

    port = get_port()[1]
    print("Port ", port)
    print("ray start --head --redis-port={}".format(port))
    os.system('ray start --head --redis-port={}'.format(port))

    temp_dir = 'temp_{}'.format('_'.join(str(time.time()).split('.')))
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    worker_file = os.path.join(temp_dir, 'worker_script')
    worker_script = create_worker_script.format(addr, port, worker_time)
    with open(worker_file, 'w') as f:
        f.write(worker_script)
    print("Worker file ", worker_file)
    for i in range(n_workers):
        print("{} {}".format(qsub_path, worker_file))
        print(list(qsub_path))
        print(list(worker_file))
        os.system('{} -l nodes=1:ppn=2 {}'.format(qsub_path, worker_file))
    # print("{} {}".format(qsub_path, worker_file))
    print("{}:{}".format(addr, port))
    ray.init(redis_address='{}:{}'.format(addr, port))
    print("Ray initialized")
    return wait_for_workers(n_workers+1)


def main():
    init_pbs_ray()

if __name__ == "__main__":
    main()



