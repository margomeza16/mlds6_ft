from numpy.typing import ArrayLike
from tensorflow import Tensor
from mldsft.data import DataLoader
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy
#from tensorflow.keras.applications import InceptionV3
from dataclasses import dataclass
from typing import Tuple

class MLP(Model):
    def __init__(self, dropout: float, *args, **kwargs):
        super(MLP, self).__init__(*args, **kwargs)
        #self.feature_extractor = InceptionV3(include_top=False)
        self.flat = Flatten()
        self.den1 = Dense(256, activation="relu")
        self.drop1 = Dropout(dropout)
        self.den2 = Dense(10, activation="softmax")

    def call(self, x: Tensor) -> Tensor:
        #h = self.feature_extractor(x)
        f = self.flat(x)
        h1 = self.den1(f)
        d1 = self.drop1(h1)
        out = self.den2(d1)
        return out

@dataclass
class CompileHParams:
    dropout: float
    learning_rate: float
    input_size: Tuple

@dataclass
class TrainHParams:
    epochs: int
    batch_size: int

class ModelBuilder:
    compile_hparams: CompileHParams
    train_hparams: TrainHParams
    model: Model
    #model: Model

    def set_compile_hparams(self, compile_hparams: CompileHParams) -> "ModelBuilder":
        self.compile_hparams = compile_hparams
        return self

    def set_train_hparams(self, train_hparams: TrainHParams) -> "ModelBuilder":
        self.train_hparams = train_hparams
        return self

    def build(self) -> "ModelBuilder":
        self.model = MLP(dropout=self.compile_hparams.dropout)
        self.model.build(self.compile_hparams.input_size)
        self.model.compile(
                loss="categorical_crossentropy",
                optimizer=Adam(learning_rate=self.compile_hparams.learning_rate),
                metrics=CategoricalAccuracy()
                )
        return self

    def train(self, dl: DataLoader) -> "ModelBuilder":
        X_train, Y_train, X_test, Y_test = dl()
        self.model.fit(
                X_train, Y_train, epochs=self.train_hparams.epochs,
                validation_data=(X_test, Y_test),
                batch_size=self.train_hparams.batch_size
                )
        return self
