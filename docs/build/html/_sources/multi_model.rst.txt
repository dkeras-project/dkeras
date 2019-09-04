Multiple Model Inference
========================

.. code-block:: python

    from tensorflow.keras.applications import ResNet50, MobileNet
    from dkeras import dKeras
    import numpy as np
    import ray

    ray.init()
    model1 = dKeras(ResNet50, weights='imagenet', wait_for_workers=True, n_workers=3)
    model2 = dKeras(MobileNet, weights='imagenet', wait_for_workers=True, n_workers=3)

    test_data = np.random.uniform(-1, 1, (100, 224, 224, 3))

    model1.predict(test_data)
    model2.predict(test_data)

    model1.close()
    model2.close()

