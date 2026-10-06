import json
from pathlib import Path
from .utils import write_json


def save_report(directory, stem, data, intro=""):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / (stem + ".json"), data)
    (directory / (stem + ".md")).write_text(
        "# "
        + stem.replace("_", " ").title()
        + "\n\n"
        + intro
        + "\n\n```json\n"
        + json.dumps(data, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )


def confusion_plots(y, pred, names, directory, prefix):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, pred, labels=list(range(len(names))))
    for normalized in [False, True]:
        a = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalized else cm
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            a,
            annot=True,
            fmt=".2f" if normalized else "d",
            xticklabels=names,
            yticklabels=names,
            ax=ax,
            cmap="Blues",
        )
        ax.set(
            xlabel="Predicted class",
            ylabel="True class",
            title=prefix.replace("_", " "),
        )
        fig.tight_layout()
        fig.savefig(
            Path(directory)
            / (
                prefix
                + "_confusion_matrix"
                + ("_normalized" if normalized else "")
                + ".png"
            ),
            dpi=160,
        )
        plt.close(fig)
    return cm.tolist()


def history_plots(logs, reports):
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = [
        pd.read_csv(p)
        for p in [
            Path(logs) / "training_history_stage1.csv",
            Path(logs) / "training_history_finetune.csv",
        ]
        if p.exists()
    ]
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    for metric in ["accuracy", "loss"]:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(df) + 1), df[metric], label="train")
        ax.plot(range(1, len(df) + 1), df["val_" + metric], label="validation")
        ax.set(xlabel="Epoch (concatenated stages)", ylabel=metric)
        ax.legend()
        fig.tight_layout()
        fig.savefig(Path(reports) / ("training_" + metric + ".png"))
        plt.close(fig)
