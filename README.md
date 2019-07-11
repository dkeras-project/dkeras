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


## Coming Soon: Installation
Just pip, no messy compilers or installing from source necessary. Machine learning can be hard enough without extra headaches.
```bash
pip install dkeras
```

Or if you want to develop using dKeras:
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
license restrictions." This will enable Keras to be run on 

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

## Distributed Training

Under construction...