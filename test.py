from neuralNetwork import NeuralNetwork , NeuralLayer, ActivationLayer
import numpy as np

class Relu(ActivationLayer):

    def function(self, value: float):
        return max(0, value)
    
    def derivative(self, value: float):
        return int( value < 0)

net = NeuralNetwork( epoch=1000)
net.add_layer(NeuralLayer(3,100))
net.add_layer(Relu())
net.add_layer(NeuralLayer(100,2))
net.add_layer(Relu())

x = np.array([
    [0,0,0],
    [0,0,1],
    [0,1,0],
    [0,1,1],
    [1,0,0],
    [1,0,1],
    [1,1,0],
    [1,1,1],
])

y = np.array([
    [1,0],
    [0,1],
    [1,0],
    [0,1],
    [1,0],
    [1,0],
    [1,0],
    [0,1],
])
net.fit(x,y)
v = net.predict(x)

print(v)