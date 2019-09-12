from tensorflow.keras.applications import ResNet50
from dkeras import dKeras
import numpy as np
import time
import ray

ray.init()

n_data = 100
data = np.random.uniform(-1, 1, (n_data, 224, 224, 3))

model = dKeras(ResNet50, init_ray=False, wait_for_workers=True, n_workers=4)

start_time = time.time()
preds = model.predict(data, int8_cvrt=True)
elapsed = time.time() - start_time
print(elapsed, n_data/elapsed)