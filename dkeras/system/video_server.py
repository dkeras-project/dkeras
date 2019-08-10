#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import print_function, division

import os
import time

import cv2
import ray


video_server_init_message = """
Initializing video server...
Video source: {}
Queue: {}
"""

@ray.remote
class VideoServer(object):

    def __init__(self,
                 server,
                 output_queues ='frame',
                 camera=None,
                 file=None,
                 stream_name=None,
                 wait=True,
                 verbose=True,
                 scale=1):
        if not isinstance(output_queues, (str, tuple, list)):
            raise TypeError(
                "output_queues arg must be str, tuple, or list, not".format(
                    type(output_queues).__name__))
        if (file is None) and (camera is None):
            raise UserWarning("Must provide either a file or camera arg")
        elif not (file is None):
            _camera_cv2id = file
        elif not (camera is None):
            _camera_cv2id = camera
        timestamp = time.ctime().replace(' ', '_')
        if stream_name is None:
            if camera is None:
                self.stream_name = 'camera_{}_{}'.format(
                    os.path.basename(file).replace('.', '_'), timestamp)
            else:
                self.stream_name = 'camera_{}_{}'.format(camera, timestamp)
        if not isinstance(scale, (int, float)):
            raise TypeError("Scale must be type float, not {}".format(
                type(scale).__name__))
        self.scale = scale
        self.server = server
        if isinstance(output_queues, str):
            self.output_queues = [output_queues]
        else:
            self.output_queues = output_queues
        if not isinstance(wait, bool):
            raise TypeError("wait arg must be boolean, not {}".format(
                type(wait).__name__))
        self.wait = wait
        self.verbose = verbose
        self.video_data = cv2.VideoCapture(_camera_cv2id)
        self.camera_width = self.video_data.get(3)
        self.camera_height = self.video_data.get(4)
        # TODO: Move this outside __init__
        if self.verbose:
            print(video_server_init_message.format(
                _camera_cv2id, self.output_queues))
        self.main()

    def main(self):
        size = (int(self.camera_width / self.scale),
                int(self.camera_height / self.scale))
        while True:
            ret, frame = self.video_data.read()
            frame = cv2.resize(frame, size)
            data = {
                "frame": frame,
                "time": time.ctime(),
                "stream_name" : self.stream_name
            }
            data = ray.put(data)
            if ret is True:
                if self.wait:
                    while not (ray.get(self.server.can_push.remote(self.output_queues[0]))):
                        time.sleep(1e-4)
                self.server.push.remote(data, self.output_queues)
            else:
                break
