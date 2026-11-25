import numpy as np
import pandas as pd
from scipy.signal import detrend, windows



def find_dominant_frequencies(x: np.ndarray, fs: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    #Remove linear trends from the signals
    x_detrended = detrend(x, axis=1)

    #Apply a Hann window to reduce spectral leakage
    window = windows.hann(x.shape[1])
    x_windowed = x_detrended * window

    #Compute the Fast Fourier Transform
    fft_values = np.fft.rfft(x_windowed, axis=1)

    #Calculate the magnitude spectrum
    magnitudes = np.abs(fft_values)

    #Generate corresponding frequency bins
    frequencies = np.fft.rfftfreq(x.shape[1], d=1/fs)

    #Find the strongest frequency component
    dominant_indices = np.argmax(magnitudes, axis=1)

    #Return the dominant frequencies
    return frequencies[dominant_indices]



def extract_features(data: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    data = np.asarray(data, dtype=float)
    labels = np.asarray(labels)

    if data.ndim != 3 or data.shape[2] != 2:
        raise ValueError("Data must have shape (samples, timesteps, 2)")

    if len(labels) != len(data):
        raise ValueError("Number of labels must match samples")

    features = {}

    # Extract features for current and voltage separately
    for channel_idx, channel_name in enumerate(["current", "voltage"]):

        # Shape: (num_samples, seq_len)
        signals = data[:, :, channel_idx]

        # Statistical features
        features[f"{channel_name}_mean"] = np.mean(signals, axis=1)
        features[f"{channel_name}_std"] = np.std(signals, axis=1)
        features[f"{channel_name}_min"] = np.min(signals, axis=1)
        features[f"{channel_name}_max"] = np.max(signals, axis=1)
        features[f"{channel_name}_median"] = np.median(signals, axis=1)

        # Signal features
        features[f"{channel_name}_peak_to_peak"] = np.ptp(signals, axis=1)
        features[f"{channel_name}_rms"] = np.sqrt(
            np.mean(signals ** 2, axis=1)
        )
        features[f"{channel_name}_variance"] = np.var(signals, axis=1)
        features[f"{channel_name}_energy"] = np.sum(signals ** 2, axis=1)

        # Frequency-domain feature
        features[f"{channel_name}_dominant_frequency"] = (
            find_dominant_frequencies(signals, fs=100)
        )

    # Convert features to a DataFrame
    features_df = pd.DataFrame(features)

    # Add labels
    features_df["labels"] = labels

    return features_df


