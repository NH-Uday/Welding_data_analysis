import numpy as np
import pandas as pd  
from pathlib import Path
import logging


def load_data(data_path: Path) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    data = remove_unlabeled_data(data).dropna().reset_index(drop=True)
    if data.empty:
        raise ValueError("No labeled rows with complete data remain.")
    return data

def remove_unlabeled_data(data: pd.DataFrame) -> pd.DataFrame:
    return data.loc[data["labels"] != -1].copy()


def convert_to_np(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = data["labels"].to_numpy()
    exp_ids = data["exp_ids"].to_numpy()
    current_columns = [col for col in data.columns if col.startswith("I")]
    voltage_columns = [col for col in data.columns if col.startswith("V")]

    if not current_columns or len(current_columns) != len(voltage_columns):
        raise ValueError("Expected equal numbers of current and voltage columns.")

    current = data[current_columns].to_numpy(dtype=float)
    voltage = data[voltage_columns].to_numpy(dtype=float)
    features = np.stack((current, voltage), axis=-1)
    return labels, exp_ids, features


def create_sliding_windows_first_dim(data: np.ndarray, sequence_length: int) -> np.ndarray:
    if data.ndim != 3:
        raise ValueError("Expected data with shape (samples, timesteps, features).")
    if not 1 <= sequence_length <= len(data):
        raise ValueError("Sequence length must be between 1 and the number of samples.")

    windows = np.lib.stride_tricks.sliding_window_view(data, sequence_length, axis=0)
    return windows.transpose(0, 3, 1, 2)

def get_welding_data(path: Path, n_samples: int | None = None, return_sequences: bool = False, sequence_length: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    np_file_path_data = path.parent / "data.npy"
    np_file_path_labels = path.parent / "labels.npy"
    np_file_path_exp_ids = path.parent / "exp_ids.npy"

    if not np_file_path_data.exists() or not np_file_path_labels.exists() or not np_file_path_exp_ids.exists():
        data = load_data(path)
        labels, exp_ids, np_data = convert_to_np(data)
        logging.info(f"Saving data to {np_file_path_data}")

        np.save(np_file_path_data, np_data)
        np.save(np_file_path_labels, labels)
        np.save(np_file_path_exp_ids, exp_ids)

    np_data = np.load(np_file_path_data)
    labels = np.load(np_file_path_labels)
    exp_ids = np.load(np_file_path_exp_ids)

    if return_sequences:
        np_data = create_sliding_windows_first_dim(np_data, sequence_length)
        labels = np.repeat(labels.reshape(-1, 1), sequence_length, axis=1)[:-sequence_length + 1]
        exp_ids = np.repeat(exp_ids.reshape(-1, 1), sequence_length, axis=1)[:-sequence_length + 1]

    if n_samples is not None:
        sample_idx = np.random.choice(np_data.shape[0], n_samples, replace=False)
        logging.info(f"Sampling {n_samples} samples from {np_data.shape[0]}")
        np_data = np_data[sample_idx]
        labels = labels[sample_idx]
        exp_ids = exp_ids[sample_idx]

    return np_data, labels, exp_ids