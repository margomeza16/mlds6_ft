from modft.data import DataLoader
import mlflow

def main() -> int:
    _, _, X_test, _ = DataLoader()()
train_gen,X_train_prep,X_val_prep,X_test_prep,Y_train,Y_test,Y_val = DataLoader()()
    samples = X_test_prep[:2]
    name = "runs:/c8df84ac830d492685122538a67aa5bc/mlp"
    model = mlflow.pyfunc.load_model(name)
    y_pred = model.predict(samples)
    print(y_pred)

    return 0

if __name__ == "__main__":
    exit(main())
