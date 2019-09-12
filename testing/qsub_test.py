from tensorflow.keras.applications import ResNet50
from dkeras import dKeras
from dkeras.utils.qsub_functions import init_pbs_ray
import numpy as np
import time
import ray

init_pbs_ray()
print(ray.nodes())

data = np.random.uniform(-1, 1, (10000, 224, 224, 3))

start_time = time.time()
model = dKeras(ResNet50, init_ray=False, wait_for_workers=True, n_workers=500)
elapsed = time.time() - start_time

print("Workers initialized after {}".format(elapsed))

start_time = time.time()
preds = model.predict(data)
elapsed = time.time() - start_time

print("Preds after {}".format(elapsed))
