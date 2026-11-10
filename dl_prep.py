"""Közös adat-előkészítés és kiértékelés a négy mélytanulásos modellhez.

Mindenki ezt importálja, így a négy DL modell és a négy klasszikus modell
eredményei ugyanazon a felosztáson, ugyanazzal a tokenizerrel mérhetők:

    import dl_prep as dp
    dp.set_seeds()
    data = dp.load_data()
    model = keras.Sequential([...])            # saját architektúra
    dp.compile_model(model)
    history, seconds = dp.fit(model, data)
    result = dp.evaluate(model, data)          # CSAK a végső modellen!
    dp.save_result("lstm", "Bidirectional LSTM", model, history, result, seconds)

Lépések (mind rögzített, ne írjátok át egyénileg):
  1. tisztítás: train_models.prepare() – 7432 pár, közel-duplikátum szűrés ≥ 0,85
  2. címkék: original → 1 (szakmai), simplified → 0 (laikus)
  3. teszt: GroupShuffleSplit cikkenként, test_size=0,2, random_state=42
  4. validáció: a train-ből még 12% cikkenként (GroupShuffleSplit, random_state=42)
  5. EGY Keras Tokenizer, csak a train szövegeken illesztve, num_words=20 000
  6. maxlen = a train mondathosszok 95. percentilise, padding/truncating="post"
     (üres token-sorozat → [<OOV>], hogy a mask_zero=True ne adjon NaN-t)
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit

from train_models import prepare

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras.preprocessing.sequence import pad_sequences  # noqa: E402
from tensorflow.keras.preprocessing.text import Tokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

# --- Közös beállítások --------------------------------------------------------
SEED = 42
NUM_WORDS = 20_000
MAXLEN_PERCENTILE = 95
VAL_SIZE = 0.12
EMBED_DIM = 100
BATCH_SIZE = 64
MAX_EPOCHS = 30
PATIENCE = 3
LABEL_NAMES = ["Laikus (0)", "Szakmai (1)"]


@dataclass
class Data:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    tokenizer: Tokenizer
    maxlen: int
    vocab_size: int  # az Embedding input_dim értéke

    def summary(self) -> str:
        return (
            f"train: {len(self.y_train)} mondat ({self.train_df.article.nunique()} cikk) | "
            f"val: {len(self.y_val)} ({self.val_df.article.nunique()} cikk) | "
            f"test: {len(self.y_test)} ({self.test_df.article.nunique()} cikk)\n"
            f"szókincs (train): {len(self.tokenizer.word_index)} szó, ebből használt: {self.vocab_size} index | "
            f"maxlen: {self.maxlen} token"
        )


def set_seeds(seed: int = SEED) -> None:
    """Python, NumPy és TensorFlow seed egyszerre (tf.random.set_seed(42) is benne van)."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def split_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A tisztított mondatok train / val / test DataFrame-jei (text, label, article)."""
    train_full, test, _ = prepare()
    train_full = train_full.reset_index(drop=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED)
    tr_idx, val_idx = next(splitter.split(train_full, groups=train_full.article))
    return (
        train_full.iloc[tr_idx].reset_index(drop=True),
        train_full.iloc[val_idx].reset_index(drop=True),
        test.reset_index(drop=True),
    )


def load_data() -> Data:
    train, val, test = split_frames()
    tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(train.text)  # csak train → nincs adatszivárgás

    def seqs(df: pd.DataFrame) -> list[list[int]]:
        # Üres sorozat (pl. a 6 db "_" simplified mondat a train-ben) → egyetlen <OOV> token.
        # Különben mask_zero=True mellett a maszk csupa False, az átlagolás 0/0 → NaN súlyok.
        oov = tokenizer.word_index[tokenizer.oov_token]
        return [s or [oov] for s in tokenizer.texts_to_sequences(df.text)]

    train_seqs = seqs(train)
    maxlen = int(np.ceil(np.percentile([len(s) for s in train_seqs], MAXLEN_PERCENTILE)))

    def pad(s: list[list[int]]) -> np.ndarray:
        return pad_sequences(s, maxlen=maxlen, padding="post", truncating="post")

    return Data(
        x_train=pad(train_seqs), y_train=train.label.to_numpy(),
        x_val=pad(seqs(val)), y_val=val.label.to_numpy(),
        x_test=pad(seqs(test)), y_test=test.label.to_numpy(),
        train_df=train, val_df=val, test_df=test,
        tokenizer=tokenizer, maxlen=maxlen,
        vocab_size=min(NUM_WORDS, len(tokenizer.word_index) + 1),
    )


# --- Tanítás ------------------------------------------------------------------
def compile_model(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def early_stopping() -> keras.callbacks.EarlyStopping:
    return keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)


def fit(model: keras.Model, data: Data, epochs: int = MAX_EPOCHS, batch_size: int = BATCH_SIZE,
        use_early_stopping: bool = True, verbose: int = 0, **kwargs):
    """Tanítás a train halmazon, validálás a val halmazon. Visszaad: (history, tanítási idő mp-ben)."""
    callbacks = list(kwargs.pop("callbacks", []))
    if use_early_stopping:
        callbacks.append(early_stopping())
    start = time.perf_counter()
    history = model.fit(
        data.x_train, data.y_train,
        validation_data=(data.x_val, data.y_val),
        epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose, **kwargs,
    )
    return history, time.perf_counter() - start


def best_epoch(history) -> int:
    """Az az epoch (1-től számozva), ahol a val_loss minimális – ezt állítja vissza az EarlyStopping."""
    return int(np.argmin(history.history["val_loss"])) + 1


def evaluate(model: keras.Model, data: Data, split: str = "test") -> dict:
    """Accuracy, F1 és konfúziós mátrix. split="val" a kísérletezéshez, "test" csak a legvégén."""
    x, y = (data.x_test, data.y_test) if split == "test" else (data.x_val, data.y_val)
    prob = model.predict(x, verbose=0).ravel()
    pred = (prob >= 0.5).astype(int)
    return {
        "split": split,
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "y_prob": prob.round(5).tolist(),
        "y_pred": pred.tolist(),
    }


# --- Ábrák --------------------------------------------------------------------
def plot_history(histories, title: str = "Tanulási görbe"):
    """Loss és accuracy epoch-onként. histories: History vagy {címke: History} szótár.
    Szaggatott vonal: legjobb val_loss epoch (ezt tartja meg a restore_best_weights).
    ×: ahol az EarlyStopping megáll (legjobb + PATIENCE epoch)."""
    import matplotlib.pyplot as plt

    if not isinstance(histories, dict):
        histories = {"modell": histories}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (label, h) in enumerate(histories.items()):
        h = getattr(h, "history", h)
        c = colors[i % len(colors)]
        epochs = np.arange(1, len(h["loss"]) + 1)
        best = int(np.argmin(h["val_loss"])) + 1
        for ax, metric in zip(axes, ["loss", "accuracy"]):
            ax.plot(epochs, h[metric], color=c, ls=":", lw=1.5, label=f"{label} – train")
            ax.plot(epochs, h[f"val_{metric}"], color=c, lw=2, label=f"{label} – val")
            ax.axvline(best, color=c, ls="--", lw=0.8, alpha=0.6)
            stop = min(best + PATIENCE, len(epochs))
            ax.plot(stop, h[f"val_{metric}"][stop - 1], marker="X", ms=9, color=c, ls="none")
    for ax, name in zip(axes, ["Loss (binary cross-entropy)", "Accuracy"]):
        ax.set_xlabel("Epoch")
        ax.set_title(name)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_confusion(cm, title: str = "Konfúziós mátrix", ax=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(np.asarray(cm), display_labels=LABEL_NAMES).plot(
        cmap="Blues", ax=ax, values_format="d", colorbar=False)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Jósolt címke")
    ax.set_ylabel("Valós címke")
    return ax


# --- Közös eredménytábla -------------------------------------------------------
def test_fingerprint(data_or_df) -> str:
    """A teszthalmaz ujjlenyomata: így az összehasonlító notebook ellenőrzi, hogy mindenki ugyanazon mért."""
    df = data_or_df.test_df if isinstance(data_or_df, Data) else data_or_df
    payload = "\n".join(f"{t}\t{l}" for t, l in zip(df.text, df.label))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def save_result(key: str, name: str, model: keras.Model, history, result: dict,
                train_seconds: float, data: Data, params: dict | None = None, notes: str = "") -> Path:
    """Egy sor a közös összehasonlító táblába: results/<key>.json.
    key: rövid azonosító (pl. "baseline", "bilstm", "cnn", "bigru"); name: a táblában megjelenő név."""
    if result.get("split") != "test":
        raise ValueError("A közös táblába a teszthalmazon mért eredmény kerül (evaluate(..., split='test')).")
    RESULTS_DIR.mkdir(exist_ok=True)
    h = getattr(history, "history", history)
    record = {
        "key": key,
        "name": name,
        "accuracy": result["accuracy"],
        "f1": result["f1"],
        "confusion_matrix": result["confusion_matrix"],
        "params": int(model.count_params()),
        "train_seconds": round(float(train_seconds), 2),
        "epochs_run": len(h["loss"]),
        "best_epoch": int(np.argmin(h["val_loss"])) + 1,
        "history": {k: [round(float(v), 5) for v in vals] for k, vals in h.items()},
        "hyperparams": params or {},
        "notes": notes,
        "setup": {"num_words": NUM_WORDS, "maxlen": data.maxlen, "embed_dim": EMBED_DIM,
                  "batch_size": BATCH_SIZE, "patience": PATIENCE, "seed": SEED,
                  "tensorflow": tf.__version__, "gpu": bool(tf.config.list_physical_devices("GPU"))},
        "test_fingerprint": test_fingerprint(data),
        "y_prob": result["y_prob"],
        "y_pred": result["y_pred"],
    }
    path = RESULTS_DIR / f"{key}.json"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_results() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS_DIR.glob("*.json"))]


if __name__ == "__main__":
    d = load_data()
    print(d.summary())
    print("Címkearány (train/val/test):", d.y_train.mean().round(3), d.y_val.mean().round(3), d.y_test.mean().round(3))
    print("Teszt ujjlenyomat:", test_fingerprint(d))
