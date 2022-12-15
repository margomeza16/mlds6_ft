from modft.data import DataLoader
import mlflow

def main() -> int:
    _,_,_,X_test_prep,_,_,_ = DataLoader()()
"""
   Función para realizar evaluación del modelo mediante la clasificación de una imagen del conjunto de test

   Return
   ------
   y_pred: porcentaje de pertenencia a cada clase asignado por el modelo a la imagen recibida
"""
    samples = X_test_prep[:2]
    name = "runs:/c8df84ac830d492685122538a67aa5bc/mlp"
    model = mlflow.pyfunc.load_model(name)
    y_pred = model.predict(samples)
    print(y_pred)

    return 0

if __name__ == "__main__":
    exit(main())
