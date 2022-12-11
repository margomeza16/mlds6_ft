from typing import Dict
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, history: Dict, path: str):
        self.history = history
        self.path = path

    def __call__(self):
        fig, axes = plt.subplots(2, 1, figsize=(7, 10))
        ax = axes[0]
        ax.set_title("Losses")
        loss_metrics = list(
                filter(lambda metric: "loss" in metric, list(self.history.keys()))
                )
        for metric in loss_metrics:
            ax.plot(self.history[metric], label=metric)
        ax.legend()
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")

        ax = axes[1]
        ax.set_title("Accuracy")
        acc_metrics = list(
                filter(lambda metric: "acc" in metric, list(self.history.keys()))
                )
        for metric in acc_metrics:
            ax.plot(self.history[metric], label=metric)
        ax.legend()
        ax.set_xlabel("Epochs")
        ax.set_ylabel("ACC")
        fig.savefig(self.path)
