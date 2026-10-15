"""Train an audio-score model and generate Dataset_Final/submission.csv.

Usage:
    "C:/Program Files/Python311/python.exe" audio_model.py
    "C:/Program Files/Python311/python.exe" audio_model.py --cv
"""

from __future__ import annotations

import argparse
import csv
import wave
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


ROOT = Path(__file__).resolve().parent / "Dataset_Final"
SAMPLE_RATE = 16_000
FRAME_LENGTH = 2_048
HOP_LENGTH = 1_024
N_MELS = 40


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM audio: {path}")
    signal = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        signal = signal.reshape(-1, channels).mean(axis=1)
    return signal, sample_rate


def hz_to_mel(frequency: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency) / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    points = np.linspace(hz_to_mel(0.0), hz_to_mel(sample_rate / 2), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(points) / sample_rate).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = bins[index : index + 3]
        if center > left:
            filters[index, left:center] = np.arange(center - left) / (center - left)
        if right > center:
            filters[index, center:right] = (right - np.arange(center, right)) / (right - center)
    return filters


def _summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    percentiles = np.percentile(values, [10, 25, 50, 75, 90], axis=0)
    result: dict[str, float] = {}
    for index, percentile in enumerate((10, 25, 50, 75, 90)):
        for column, value in enumerate(percentiles[index]):
            result[f"{prefix}{column}_p{percentile}"] = float(value)
    for column, value in enumerate(np.mean(values, axis=0)):
        result[f"{prefix}{column}_mean"] = float(value)
    for column, value in enumerate(np.std(values, axis=0)):
        result[f"{prefix}{column}_std"] = float(value)
    return result


def extract_features(path: Path) -> dict[str, float]:
    signal, sample_rate = read_audio(path)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sample_rate}: {path}")

    if signal.size < FRAME_LENGTH:
        signal = np.pad(signal, (0, FRAME_LENGTH - signal.size))
    frame_count = 1 + (signal.size - FRAME_LENGTH) // HOP_LENGTH
    frames = np.lib.stride_tricks.sliding_window_view(signal, FRAME_LENGTH)[
        ::HOP_LENGTH
    ][:frame_count]
    windowed = frames * np.hanning(FRAME_LENGTH)
    spectrum = np.abs(np.fft.rfft(windowed, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(FRAME_LENGTH, 1.0 / sample_rate)
    magnitude = np.sqrt(spectrum)
    energy = np.maximum(spectrum.sum(axis=1), 1e-12)
    normalized_spectrum = spectrum / energy[:, None]

    mel = mel_filterbank(sample_rate, FRAME_LENGTH, N_MELS)
    log_mel = np.log(np.maximum(spectrum @ mel.T, 1e-10))
    spectral_centroid = (normalized_spectrum * frequencies).sum(axis=1)
    spectral_bandwidth = np.sqrt(
        (normalized_spectrum * (frequencies - spectral_centroid[:, None]) ** 2).sum(axis=1)
    )
    cumulative = np.cumsum(normalized_spectrum, axis=1)
    rolloff = frequencies[np.argmax(cumulative >= 0.85, axis=1)]
    flatness = np.exp(np.mean(np.log(np.maximum(magnitude, 1e-10)), axis=1)) / (
        np.mean(magnitude, axis=1) + 1e-10
    )
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    zero_crossings = np.mean(
        np.abs(np.diff(np.signbit(frames), axis=1)), axis=1
    )

    features: dict[str, float] = {
        "duration_seconds": float(signal.size / sample_rate),
        "global_rms": float(np.sqrt(np.mean(signal**2) + 1e-12)),
        "global_abs_mean": float(np.mean(np.abs(signal))),
        "global_peak": float(np.max(np.abs(signal))),
        "silence_fraction": float(np.mean(np.abs(signal) < 0.01)),
    }
    features.update(_summary(log_mel, "mel_"))
    features.update(_summary(np.column_stack(
        (rms, spectral_centroid, spectral_bandwidth, rolloff, flatness, zero_crossings)
    ), "spectral_"))
    return features


def load_split(split: str) -> tuple[list[str], np.ndarray | None, np.ndarray]:
    rows = list(csv.DictReader((ROOT / f"{split}.csv").open(newline="")))
    filenames = [row["filename"] for row in rows]
    labels = None if split == "test" else np.array([float(row["label"]) for row in rows])
    features = np.array(
        [[value for value in extract_features(ROOT / split / filename).values()]
         for filename in filenames],
        dtype=np.float32,
    )
    return filenames, labels, features


def make_model() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )


def quantize_predictions(predictions: np.ndarray) -> np.ndarray:
    return np.clip(predictions, 0.0, 5.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", action="store_true", help="report 5-fold validation metrics")
    parser.add_argument("--output", type=Path, default=ROOT / "submission.csv")
    args = parser.parse_args()

    train_names, labels, train_features = load_split("train")
    test_names, _, test_features = load_split("test")
    assert labels is not None

    if args.cv:
        stratification = labels.astype(str)
        fold_scores: list[tuple[float, float]] = []
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, valid_index in splitter.split(train_features, stratification):
            model = make_model()
            model.fit(train_features[train_index], labels[train_index])
            raw = model.predict(train_features[valid_index])
            predictions = quantize_predictions(raw)
            fold_scores.append((
                mean_absolute_error(labels[valid_index], predictions),
                root_mean_squared_error(labels[valid_index], predictions),
            ))
        scores = np.array(fold_scores)
        print(f"CV MAE: {scores[:, 0].mean():.4f} +/- {scores[:, 0].std():.4f}")
        print(f"CV RMSE: {scores[:, 1].mean():.4f} +/- {scores[:, 1].std():.4f}")

    model = make_model()
    model.fit(train_features, labels)
    predictions = quantize_predictions(model.predict(test_features))
    with args.output.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["filename", "label"])
        writer.writerows(zip(test_names, predictions))
    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
