Basic Inference Example
=======================

dKeras maintains the same API as Keras, the only differences are the addition
of functions to maintain distributed systems and arguements.

.. code-block:: python

    from tensorflow.keras.applications import ResNet50
    from dkeras import dKeras
    import numpy as np

    data = np.random.uniform(-1, 1, (100, 224, 224, 3))

    model = dKeras(ResNet50, wait_for_workers=True, n_workers=4)
    preds = model.predict(data)
