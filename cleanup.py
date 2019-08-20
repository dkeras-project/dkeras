#!/bin/env/python
#-*- encoding: utf-8 -*-
"""

"""
from __future__ import print_function, division
import os

def main():
    # OSX Cleanup
    dirpath = os.path.dirname(os.path.abspath(__file__))
    for root, dirs, files in os.walk(dirpath):
        for d in dirs:
            if d in ['__pycache__']:
                path = os.path.join(root, d)

                print("Removing: {}".format(path))
                os.system('rm -rf {}'.format(path))

    print("Removing pype.egg-info")
    os.system('rm -rf {}'.format(os.path.join(dirpath,'pype.egg-info')))


if __name__ == "__main__":
    main()