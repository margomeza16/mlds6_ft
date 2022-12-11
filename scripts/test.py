from modft.data import DataLoader
import mlflow

def main() -> int:
    _, _, X_test, _ = DataLoader()()
    samples = X_test[:2]
    name = "runs:/c8df84ac830d492685122538a67aa5bc/mlp"
    model = mlflow.pyfunc.load_model(name)
    y_pred = model.predict(samples)
    print(y_pred)
    
    return 0

if __name__ == "__main__":
    exit(main())
