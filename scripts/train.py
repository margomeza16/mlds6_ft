import mlflow
from modft.models import ModelBuilder, CompileHParams, TrainHParams
from modft.data import DataLoader
from modft.viz import Visualizer
from argparse import ArgumentParser

def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    return parser

def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    compile_hparams = CompileHParams(
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            input_size=(None,224,224,3)
#            input_size=(None, 784)
            )
    train_hparams = TrainHParams(
            epochs=args.epochs,
            batch_size=args.batch_size,
            )
    dl = DataLoader()

    with mlflow.start_run():
        builder = (
                ModelBuilder()
                .set_compile_hparams(compile_hparams)
                .set_train_hparams(train_hparams)
                .build()
                .train(dl)
                )
        mlflow.tensorflow.log_model(
                builder.model,
                "mlp"
                )
        history = builder.model.history.history
        Visualizer(history, "metrics.png")()
        last_metrics = {metric: vals[-1] for metric, vals in history.items()}

        mlflow.log_params({
            "dropout": compile_hparams.dropout, "learning_rate": compile_hparams.learning_rate,
            "epochs": train_hparams.epochs, "batch_size": train_hparams.batch_size
            })
        mlflow.log_metrics(last_metrics)
        mlflow.log_artifact("metrics.png", "metrics")
    return 0

if __name__ == "__main__":
    exit(main())
