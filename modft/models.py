from numpy.typing import ArrayLike
from tensorflow import Tensor
from tensorflow.keras import layers 
from modft.data import DataLoader
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.applications import ResNet50V2
from dataclasses import dataclass
from typing import Tuple

class MLP(Model):
    def __init__(self, dropout: float, *args, **kwargs):
        super(MLP, self).__init__(*args, **kwargs)
        self.feature_extractor = ResNet50V2(weights='imagenet',include_top=False,input_shape=(224, 224, 3))
        for layer in self.feature_extractor.layers:
	    layer.trainable=False
        self.pool = GlobalAveragePooling2D()
        self.den1 = Dense(32, activation="relu")
        self.drop1 = Dropout(dropout)
        self.den2 = Dense(4, activation="softmax")

    def call(self, x: Tensor) -> Tensor:
        fe = self.feature_extractor.output
        p = self.pool(fe)
        h1 = self.den1(p)
        d1 = self.drop1(h1)
        h2 = self.den2(d1)
	tl_mod = tf.keras.models.Model(inputs=[self.feature_extractor.input], outputs=[h2])
        return tl_mod

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
#        X_train, Y_train, X_test, Y_test = dl()
        train_gen, X_train_prep, X_val_prep, X_test_prep, Y_train,Y_test, Y_val = d1()
        epochs = self.train_hparams.epochs
        batch_size = self.train_hparams.batch_size
        steps_per_epoch = X_train_prep.shape[0]//batch_size
        self.model.fit(
                train_gen, epochs=epochs,
                validation_data=(X_val_prep, Y_val),
                batch_size=batch_size,
          	steps_per_epoch=steps_per_epoch
                )
        return self
