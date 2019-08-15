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
    return subprocess.check_output(['which', 'qsub']).decode()


create_worker_script = """
ray start --redis-address={}:{}
sleep {}
"""

@ray.remote
def _get_node_IPs():
    time.sleep(1e-4)
    return ray.services.get_node_ip_address()

def wait_for_IPs(n_ips, timeout=300):
    start_time = time.time()
    while True:
        ips = set(ray.get([_get_node_IPs.remote() for _ in range(1000)]))
        print(ips)
        print(len(ips))
        time.sleep(1)
        if len(ips) >= n_ips:
            return True
        if (start_time-time.time() >= timeout):
            return False


def init_pbs_ray(n_workers=3, rm_existing=True, iface_name='eno1', worker_time=3600):
    if ray.is_initialized():
        if rm_existing:
            ray.shutdown()
    qsub_path = _which_qsub()
    addresses = get_addr()
    addr = addresses[0]
    if addr == 'No IP addr':
        raise Exception("Address not found for {}".format(iface_name))

    port = get_port()
    os.system('ray start --head --redis-port={}'.format(port))

    temp_dir = 'temp_{}'.format('_'.join(str(time.time()).split('.')))
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    worker_file = os.path.join(temp_dir, 'worker_script.sh')
    worker_script = create_worker_script.format(addr, port, worker_time)
    with open(worker_file, 'w') as f:
        f.write(worker_script)

    for i in range(n_workers):
        os.system('{} {}'.format(qsub_path, worker_file))

    ray.init(redis_address='{}:{}'.format(addr, port))
    return wait_for_IPs(n_workers+1)


def main()
    init_pbs_ray()

if __name__ == "__main__":
    main()



