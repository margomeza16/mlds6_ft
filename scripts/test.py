from modft.data import DataLoader
import mlflow

def main() -> float:
    _,_,_,X_test_prep,_,_,_ = DataLoader()()
"""
   Función para realizar evaluación del modelo entrenado y cargado en mlflow  mediante la clasificación de una imagen del conjunto de test

   Return
   ------
   y_pred: probabilidad de pertenencia a cada clase asignada por el modelo a la imagen cargada
"""

    samples = X_test_prep[:2]
    name = "runs:/c8df84ac830d492685122538a67aa5bc/mlp"
    model = mlflow.pyfunc.load_model(name)
    y_pred = model.predict(samples)

    return y_pred

if __name__ == "__main__":
    exit(main())
