from tensorflow.keras.applications import ResNet50
from dkeras import dKeras
import numpy as np
import time
import ray

ray.init()
data = np.random.uniform(-1, 1, (1000, 224, 224, 3))

for i in range(1, 51):
    model = dKeras(ResNet50, init_ray=False, wait_for_workers=True, n_workers=i)

    start_time = time.time()
    preds = model.predict(data)
    elapsed = time.time() - start_time

    model.close()
    print("{}\n{} Workers\tTime: {}\tFPS: {}".format('-' * 80, i, elapsed, 1000 / elapsed))
    time.sleep(3)
