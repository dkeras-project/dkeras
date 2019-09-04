from tensorflow.keras.applications import ResNet50
from dkeras import dKeras
import numpy as np
import ray

ray.init()

data = np.random.uniform(-1, 1, (100, 224, 224, 3))

model = dKeras(ResNet50, init_ray=False, wait_for_workers=True, n_workers=4)
preds = model.predict(data)
ray.timeline('test')