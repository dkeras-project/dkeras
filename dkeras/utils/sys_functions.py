#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import division, print_function

import socket

from netifaces import AF_INET, ifaddresses, interfaces


def get_port():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(('', 0))
    addr, port = tcp.getsockname()
    tcp.close()
    return addr, port


def get_addr(iface_name):
    for ifaceName in interfaces():
        addresses = [i['addr'] for i in
            ifaddresses(ifaceName).setdefault(AF_INET,
                                              [{'addr': 'No IP addr'}])]
        if ifaceName == iface_name:
            return addresses

