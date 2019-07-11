<p align="center">
  <img src="https://github.com/gndctrl2mjrtm/dkeras/blob/master/assets/dkeras_logo.png?raw=true" alt="dKeras logo"/>
</p>

# dKeras: Distributed Keras Engine
### ***Make Keras faster with only one line of code.***

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