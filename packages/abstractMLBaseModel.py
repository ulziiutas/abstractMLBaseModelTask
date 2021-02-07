from abc import ABC, abstractmethod

class abstractMLBaseModel(ABC):        
       
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def predict(self, data):
        pass