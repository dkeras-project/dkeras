# Ideas
"""
server = dkeras.DataServer()

model1.link(model3)

model1.postprocess = lambda z: np.float16(z)

server = model1 + model2 + model3

server.add(camera1, dest=('m1', 'm2'))

server.add_address('192.168.1.42')
"""



