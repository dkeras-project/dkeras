<p align="center">
  <img src="https://github.com/gndctrl2mjrtm/dkeras/blob/master/assets/dkeras_logo.png?raw=true" alt="dKeras logo"/>
</p>

# dKeras: Distributed Keras Engine
### ***Make Keras faster with only one line of code.***

dKeras is a distributed Keras engine that is built on top of 
[Ray](https://github.com/ray-project/ray). By wrapping dKeras around your
original Keras model, it allows you to use many distributed deep learning
techniques to automatically improve your system's performance.


With an easy-to-use API and a backend framework that can be deployed from
the laptop to the data center, dKeras simpilifies what used to be a complex
and time-consuming process into only a few adjustments.

#### Why Use dKeras?

Distributed deep learning can be essential for production systems where you 
need fast inference but don't want expensive hardware accelerators or when
researchers need to train large models made up of distributable parts.

This becomes a challenge for developers because they'll need expertise in not
only deep learning but also distributed systems. A production team might also
need a machine learning optimization engineer to use neural network 
optimizers in terms of precision changes, layer fusing, or other techniques. 

Distributed inference is a simple way to get better inference FPS. The graph 
below shows how non-optimized, out-of-box models from default frameworks can 
be quickly sped up through data parallelism:

<p align="center">
  <img src="https://github.com/gndctrl2mjrtm/dkeras/blob/master/assets/inference_comparison.png?raw=true" alt="dKeras graph"/>
</p>


#### Current Capabilities:
- Data Parallelism Inference

#### Future Capabilities:
- Model Parallelism Inference
- Distributed Training
- Easy Multi-model production-ready building
- Data stream input distributed inference
- PlaidML Support
- Autoscaling
- Automatic optimal hardware configuration 
- PBS/Torque support

## Installation
dKeras will be available with the first official release coming soon. For 
now, install from source.
```bash
git clone https://github.com/gndctrl2mjrtm/dkeras
cd dkeras
pip install -e .
```

### Requirements

- Python 3.6 or higher
- ray
- psutil
- Linux (or OSX, dKeras works on laptops too!)
- numpy


### Coming Soon: [PlaidML](https://github.com/plaidml/plaidml) Support
dKeras will soon work alongside with [PlaidML](https://github.com/plaidml/plaidml), 
a "portable tensor compiler for enabling deep learning on laptops, embedded devices, 
or other devices where the available computing hardware is not well 
supported or the available software stack contains unpalatable 
license restrictions." 

## Distributed Inference

### Example

#### Original
```python
model = ResNet50()
model.predict(data)
```
#### dKeras Version
```python
from dkeras import dKeras

model = dKeras(ResNet50)
model.predict(data)
```