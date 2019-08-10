#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import print_function, division

class ModelAPI(object):

    def __init__(self, model):
        self.model = model()
        self.__dict__.update(self.model.__dict__)
        for k in dir(self.model):
            try:
                if not k in dir(self):
                    self.__dict__[k] = getattr(self.model, k)
            except AttributeError:
                pass


    def predict(self):
        print("New predict function")

def main():
    from tensorflow.keras.applications import ResNet50
    # model = ResNet50()
    model = ModelAPI(ResNet50)
    model.predict()
    model.summary()


if __name__ == "__main__":
    main()

