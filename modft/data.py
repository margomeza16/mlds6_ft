from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

class DataLoader:
    def __init__(self):
        self.loader = mnist.load_data

    def __call__(self):
        (X_train, y_train), (X_test, y_test) = self.loader()
        Y_train = to_categorical(y_train)
        Y_test = to_categorical(y_test)
        return X_train, Y_train, X_test, Y_test

# class DataLoaderAug:
#     def __init__(self):
#         self.loader = mnist.load_data
# 
#     def __next__(self):
#         (X_train, y_train), (X_test, y_test) = self.loader()
#         ... # dataaug
#         Y_train = to_categorical(y_train)
#         Y_test = to_categorical(y_test)
#         return X_train, Y_train, X_test, Y_test
