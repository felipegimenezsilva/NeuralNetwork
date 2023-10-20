import numpy as np

class __Dictionable__:
    
    def dict(self):
        raise Exception("dict function not implemented")

class __GenericLayer__(__Dictionable__):

    def forward(self, x:np.array) -> np.array:
        raise Exception('forward function not implemented')

    def backward(self, errors:np.array, learning_rate:float) -> np.array:
        raise Exception('backward function not implemented')

class NeuralLayer(__GenericLayer__):
    
    def __init__(self, fin:int, fout:int):
        self.fin, self.fout = fin, fout
        self.weigth, self.bias = None, None
        self.init_weigth()

    def init_weigth(self):
        self.weigth = np.random.uniform(-0.5,0.5,(self.fin,self.fout))
        self.bias = np.zeros((1,self.fout))

    def forward(self, x: np.array) -> np.array:
        return x @ self.weigth + self.bias
    
    def backward(self, output_error: np.array, learning_rate: float) -> np.array:
        input_error = output_error @ self.weigth.T
        weight_error = input_error @ self.weigth 
        self.bias -= learning_rate * output_error
        self.weigth -= learning_rate * weight_error
        return input_error
    
    def dict(self):
        return super().dict()

class NeuralCustomLayer(NeuralLayer):
    
    def __init__(self, fin: int, fout: int, weigth: np.array, bias: np.array):
        self.fin, self.fout = fin, fout
        self.weigth, self.bias = weigth, bias

class ActivationLayer(__GenericLayer__):

    def function(self, value: float):
        raise Exception('Activation function not implemented')

    def derivative(self, value: float):
        raise Exception('Derivative function not implemented')

    def forward(self, x: np.array) -> np.array:
        self.value = x
        return np.array([[self.function(y) for y in value] for value in x])

    def backward(self, errors: np.array, learning_rate: float) -> np.array:
        return np.array([[self.derivative(y) for y in x] for x in self.value]) * errors
    
    def dict(self):
        return super().dict()
    
class NeuralNetwork(__Dictionable__):
    
    def __init__(self, epoch : int = 100, learning_rate : float = 0.01):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.layers=[]

    def add_layer(self,layer: __GenericLayer__):
        self.layers.append(layer)

    def fit(self, x : np.array, y : np.array):
        for _ in range(self.epoch):
            for x_,y_ in zip(x,y):
                for layer in self.layers:
                    x_ = layer.forward(x_)
                error = x_ - y_
                for layer in reversed(self.layers):
                    error = layer.backward(error,self.learning_rate)
    
    def predict(self, x : np.array) -> np.array:
        for layer in self.layers: x = layer.forward(x)
        return x
