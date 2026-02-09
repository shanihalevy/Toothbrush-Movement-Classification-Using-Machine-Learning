# main_05_train_and_prepare.py

import os
import pandas as pd
import numpy as np
import zipfile
import tempfile
from scipy.stats import spearmanr, skew, kurtosis
from scipy.signal import welch, butter, filtfilt, hilbert, find_peaks
from sklearn.metrics import mutual_info_score, accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
from skrebate import ReliefF
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
import matplotlib
import joblib  # For saving and loading the model
import json  # For saving and loading dictionaries/lists

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# --- Set a global NumPy random seed for reproducibility ---
np.random.seed(42)

# Global group number for output files
GROUP_NUMBER = "05"


# --- Utility functions for saving/loading ---
def save_numpy_to_csv(data, filename, output_dir):
    pd.DataFrame(data).to_csv(os.path.join(output_dir, filename), index=False) # <--- השתמש ב-os.path.join כאן


def load_numpy_from_csv(filename, output_dir):
    return pd.read_csv(os.path.join(output_dir, filename)).values




def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


########################## Part A: Segmentation (with 65% label threshold) ##########################
def segment_signal(data_path, window_size):
    """
    Segments signal data into windows and assigns labels based on majority rule (>65% threshold).
    Windows are created only up to the last labeled time in the label table, ignoring any signal data beyond that.

    MODIFICATION: If a label file is not found, all windows for that recording will be labeled as -1.

    Args:
        data_path (str): Path to the group directory containing signal and label CSV files.
        window_size (float): Size of each window in seconds.

    Returns:
        X_matrix (np.array): Matrix of window metadata [start_time, end_time, hand_value, group_value, recording_num].
        Y_vector (np.array): Vector of labels for each window.
    """
    group_name = os.path.basename(data_path)
    group_value = int(group_name)

    all_files = os.listdir(data_path)
    signal_files = [f for f in all_files if
                    f.endswith('.csv') and ('Acc' in f or 'Gyro' in f or 'Mag' in f) and 'label' not in f]

    # Group signal files by recording and hand (e.g., 09_14_R)
    recording_groups = {}
    for fileName in signal_files:
        parts = fileName.split('_')
        recording_num = int(parts[1])
        hand = parts[2]
        sensor = parts[3].replace('.csv', '')
        key = (recording_num, hand)
        if key not in recording_groups:
            recording_groups[key] = {}
        recording_groups[key][sensor] = fileName

    X_matrix = []
    Y_vector = []

    # Track total undefined and total windows for the group
    total_undefined_count = 0
    total_window_count = 0

    for (recording_num, hand), sensors in recording_groups.items():
        # Check if all sensors (Acc, Gyro, Mag) are present
        if not all(sensor in sensors for sensor in ['Acc', 'Gyro', 'Mag']):
            print(f"Skipping recording {recording_num}_{hand} due to missing sensor files: {sensors.keys()}")
            continue

        # Load all sensor files to determine their time ranges
        time_ranges = []
        for sensor in ['Acc', 'Gyro', 'Mag']:
            fileName = sensors[sensor]
            filePath = os.path.join(data_path, fileName)
            data = pd.read_csv(filePath)
            time_col = [col for col in data.columns if 'elapsed' in col][0]
            time = data[time_col].values
            time_ranges.append((time.min(), time.max()))

        # Determine the common time range of the signal files (max of start times, min of end times)
        signal_start_time = max(tr[0] for tr in time_ranges)
        signal_end_time = min(tr[1] for tr in time_ranges)

        if signal_end_time <= signal_start_time:
            print(f"Skipping recording {recording_num}_{hand} due to non-overlapping time ranges: {time_ranges}")
            continue

        hand_value = 1 if hand == 'R' else 0

        # Load label file
        label_file_name = f"{group_name}_{recording_num:02d}_{hand}_label.csv"
        label_file_path = os.path.join(data_path, label_file_name)

        label_table = pd.DataFrame()
        label_file_found = False

        if os.path.isfile(label_file_path):
            try:
                label_table = pd.read_csv(label_file_path)
                start_col = next((col for col in label_table.columns if 'start' in col.lower()), None)
                end_col = next((col for col in label_table.columns if 'end' in col.lower()), None)
                label_col = next((col for col in label_table.columns if 'label' in col.lower()), None)

                if start_col is None or end_col is None or label_col is None:
                    print(
                        f"Error: Required columns containing 'start', 'end', or 'label' not found in {label_file_name}. All windows will be labeled -1.")
                    label_file_found = False
                else:
                    label_file_found = True
            except Exception as e:
                print(f"Error reading label file {label_file_name}: {str(e)}. All windows will be labeled -1.")
                label_file_found = False
        else:
            # print(f"Warning: Label file not found for recording {recording_num}. All windows will be labeled -1.")
            label_file_found = False

        # Determine the time range for segmentation
        if not label_file_found:
            start_time_for_segmentation = signal_start_time
            end_time_for_segmentation = signal_end_time
        else:
            label_start_time = label_table[start_col].min()
            label_end_time = label_table[end_col].max()
            start_time_for_segmentation = max(signal_start_time, label_start_time)
            end_time_for_segmentation = min(signal_end_time, label_end_time)

        if end_time_for_segmentation <= start_time_for_segmentation:
            print(
                f"Skipping recording {recording_num}_{hand} due to non-overlapping label and signal time ranges (or empty range if no label file): "
                f"signal ({signal_start_time}, {signal_end_time}), segmentation range ({start_time_for_segmentation}, {end_time_for_segmentation})")
            continue

        # Segment the signal within the determined time range
        curr_start = start_time_for_segmentation

        while curr_start + window_size <= end_time_for_segmentation + 1e-6:
            curr_end = curr_start + window_size

            assigned_label = -1

            if label_file_found:
                overlapping_labels_mask = (label_table[start_col] < curr_end - 1e-9) & \
                                          (label_table[end_col] > curr_start + 1e-9)
                overlapping_label_entries = label_table[overlapping_labels_mask]

                if not overlapping_label_entries.empty:
                    label_durations = {}

                    for _, row in overlapping_label_entries.iterrows():
                        overlap_start = max(curr_start, row[start_col])
                        overlap_end = min(curr_end, row[end_col])

                        duration = overlap_end - overlap_start

                        if duration > 1e-9:
                            label = row[label_col]
                            label_durations[label] = label_durations.get(label, 0) + duration

                    if label_durations:
                        total_duration_in_window = sum(label_durations.values())

                        if total_duration_in_window > 1e-9:
                            max_percentage = 0.0

                            for label, duration in label_durations.items():
                                percentage = duration / total_duration_in_window
                                if percentage > max_percentage:
                                    max_percentage = percentage
                                    assigned_label = label

                            if max_percentage <= 0.65:
                                assigned_label = -1

            if assigned_label == -1:
                total_undefined_count += 1

            X_matrix.append([curr_start, curr_end, hand_value, group_value, recording_num])
            Y_vector.append(assigned_label)

            curr_start += window_size
            total_window_count += 1

    X_matrix = np.array(X_matrix)
    Y_vector = np.array(Y_vector)

    # Calculate and print the percentage of unlabeled windows for the group
    if total_window_count > 0:
        unlabeled_percentage = (total_undefined_count / total_window_count) * 100
        print(
            f"Group {group_name}, window size {window_size} seconds: {unlabeled_percentage:.2f}% of windows are unlabeled "
            f"({total_undefined_count} out of {total_window_count} windows)")
    else:
        print(f"Group {group_name}, window_size {window_size} seconds: No windows to label.")

    return X_matrix, Y_vector


##################### Part B: Feature Extraction ######################################

def butter_lowpass_filter(data, cutoff=5, fs=50, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def calculate_thd(signal, sampling_rate):
    """
    Calculates the Total Harmonic Distortion (THD) of a signal.
    Removes DC component and finds fundamental and harmonic powers.
    """
    signal = signal - np.mean(signal)
    nperseg = 16

    freqs, psd = welch(signal, fs=sampling_rate, nperseg=nperseg)

    # Find the fundamental frequency (frequency with the highest power)
    if len(psd) == 0 or np.sum(psd) == 0:
        return 0

    fundamental_freq_index = np.argmax(psd)
    fundamental_freq = freqs[fundamental_freq_index]
    fundamental_power = psd[fundamental_freq_index]

    if fundamental_power <= 0:
        return 0

    harmonic_powers_sum = 0
    # Look for harmonics: integer multiples of the fundamental frequency (excluding 1x - the fundamental itself)
    # Start from the second harmonic (2 * fundamental_freq)
    for h in range(2, 6):
        target_freq = h * fundamental_freq
        # Find the index of the frequency closest to the harmonic
        # Ensure this frequency is within the range of frequencies calculated by Welch
        if target_freq < freqs.max():
            closest_freq_index = np.argmin(np.abs(freqs - target_freq))
            # Ensure the found frequency is sufficiently close to the expected harmonic (e.g., within a certain bandwidth)
            # This can prevent including noise if there is no clear harmonic
            if np.abs(freqs[closest_freq_index] - target_freq) < (freqs[1] - freqs[0]) * 1.5:
                harmonic_powers_sum += psd[closest_freq_index]
        else:
            break

    thd = np.sqrt(harmonic_powers_sum) / np.sqrt(fundamental_power)
    return thd


def calculate_spectral_entropy(signal, sampling_rate, nperseg=16):
    freqs, psd = welch(signal, fs=sampling_rate, nperseg=nperseg)
    mask = (freqs >= 0.5) & (freqs <= 5)
    psd = psd[mask] if np.any(mask) else psd
    psd_sum = np.sum(psd)
    if psd_sum <= 0:
        return 0
    psd_norm = psd / psd_sum
    return -np.sum(psd_norm * np.log2(psd_norm + 1e-10))


def zero_crossings(signal):
    return ((signal[:-1] * signal[1:]) < 0).sum()


def zero_crossing_variance(signal):
    zero_cross_indices = np.where(np.diff(np.signbit(signal)))[0]
    if len(zero_cross_indices) < 2:
        return 0
    intervals = np.diff(zero_cross_indices)
    return np.var(intervals)


def cycle_count(signal, sampling_rate, method='peaks'):
    if method == 'peaks':
        peaks, _ = find_peaks(signal, distance=sampling_rate // 5)
        return len(peaks) - 1 if len(peaks) > 1 else 0
    return 0


def calculate_envelope_stats(signal):
    envelope = np.abs(hilbert(signal))
    return np.mean(envelope)


def calculate_slope(signal):
    """
    Calculate the mean of the first derivative (slope) of the signal.
    """
    diff = np.diff(signal)
    slope_mean = np.mean(diff) if len(diff) > 0 else 0
    return slope_mean


def calculate_motion_trajectories(signal, sampling_rate):
    """
    Calculate cumulative displacement (trajectory) by integrating the signal.
    Assumes signal is acceleration (Acc) or angular velocity (Gyro).
    For Acc, double integration gives displacement; for Gyro, single integration gives angle.
    Returns cumulative displacement/angle in the signal.
    """
    # Remove mean to reduce drift
    signal = signal - np.mean(signal)
    # First integration (velocity for Acc, angle for Gyro)
    cumsum = np.cumsum(signal) / sampling_rate
    if len(cumsum) > 0:
        return cumsum[-1]  # Return the final cumulative value
    return 0


def calculate_cpi(signal):
    """
    Calculate Chronological Pattern Indexing (CPI) by discretizing the signal into states
    (positive, negative, zero velocity) and counting transition frequencies.
    Returns frequencies of transitions: pos-to-neg.
    """
    # Discretize signal into states based on sign
    states = np.zeros_like(signal, dtype=int)
    states[signal > 0] = 1
    states[signal < 0] = -1
    states[signal == 0] = 0

    # Count transitions
    pos_to_neg = 0
    for i in range(len(states) - 1):
        if states[i] == 1 and states[i + 1] == -1:
            pos_to_neg += 1

    return pos_to_neg


def calculate_cycle_consistency(signal, sampling_rate, min_peaks=3):
    """
    Calculate cycle consistency: cycle symmetry (positive/negative phase ratio).
    Returns average symmetry.
    """
    signal = butter_lowpass_filter(signal, cutoff=5, fs=sampling_rate)
    peaks, _ = find_peaks(signal, distance=sampling_rate // 5)
    if len(peaks) < min_peaks:
        return 0
    symmetry_ratios = []
    for i in range(len(peaks) - 1):
        start_idx = peaks[i]
        end_idx = peaks[i + 1]
        cycle_signal = signal[start_idx:end_idx]
        pos_duration = np.sum(cycle_signal > 0)
        neg_duration = np.sum(cycle_signal < 0)
        if neg_duration > 0:
            symmetry = pos_duration / neg_duration
            symmetry_ratios.append(symmetry)
        else:
            symmetry_ratios.append(0)

    avg_symmetry = np.mean(symmetry_ratios) if len(symmetry_ratios) > 0 else 0
    return avg_symmetry


def calculate_directional_transitions(signal):
    """
    Calculate the number of directional transitions (positive-to-negative or vice versa).
    Returns number of transitions.
    """
    # Discretize signal into states based on sign
    states = np.zeros_like(signal, dtype=int)
    states[signal > 0] = 1
    states[signal < 0] = -1

    # Find transitions
    transitions = 0
    current_state = states[0]
    for i in range(1, len(states)):
        if states[i] != current_state and states[i] != 0 and current_state != 0:
            transitions += 1
            current_state = states[i]

    num_transitions = transitions
    return num_transitions


def calculate_harmonic_ratios(freqs, psd, fmin=0.5, fmax=5):
    """
    Calculate the ratio of the second harmonic power to the fundamental frequency power.
    Returns second harmonic ratio.
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0
    freqs_masked = freqs[mask]
    psd_masked = psd[mask]
    fundamental_idx = np.argmax(psd_masked)
    fundamental_freq = freqs_masked[fundamental_idx]
    fundamental_power = psd_masked[fundamental_idx]

    if fundamental_power <= 0:
        return 0

    second_harmonic_freq = 2 * fundamental_freq
    if second_harmonic_freq > fmax:
        second_harmonic_power = 0
    else:
        second_harmonic_idx = np.argmin(np.abs(freqs_masked - second_harmonic_freq))
        second_harmonic_power = psd_masked[second_harmonic_idx]

    second_ratio = second_harmonic_power / fundamental_power if fundamental_power > 0 else 0
    return second_ratio


def calculate_low_freq_power(psd, freqs, fmin=0, fmax=0.5):
    """
    Calculate power in the low-frequency band (0-0.5 Hz) to detect rest states.
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0
    power = np.sum(psd[mask])
    return power


def calculate_skewness(signal):
    """
    Calculate the skewness (asymmetry) of the signal distribution.
    """
    return skew(signal, bias=False) if len(signal) > 0 else 0


def calculate_kurtosis(signal):
    """
    Calculate the kurtosis (tailedness) of the signal distribution.
    """
    return kurtosis(signal, bias=False) if len(signal) > 0 else 0


def calculate_waveform_length(signal):
    """
    Calculate Waveform Length (WL) as the sum of absolute differences between consecutive samples.
    """
    if len(signal) < 2:
        return 0
    return np.sum(np.abs(np.diff(signal)))


def calculate_largest_lyapunov_exponent(signal, time_delay=1, dimension=3, n_neighbors=5):
    """
    Calculate the Largest Lyapunov Exponent (LLE) for a signal.
    A simplified implementation using nearest neighbors to estimate divergence.

    Args:
        signal (np.array): The input signal.
        time_delay (int): Delay for embedding (default=1).
        dimension (int): Embedding dimension (default=3).
        n_neighbors (int): Number of nearest neighbors to consider (default=5).

    Returns:
        float: Estimated LLE value.
    """
    # Step 1: Phase Space Reconstruction (Takens' Embedding Theorem)
    N = len(signal)
    if N < (dimension * time_delay + n_neighbors):
        return 0  # Signal too short for meaningful LLE computation

    # Construct the phase space
    embedded_points = []
    for i in range(N - (dimension - 1) * time_delay):
        point = [signal[i + j * time_delay] for j in range(dimension)]
        embedded_points.append(point)
    embedded_points = np.array(embedded_points)

    if len(embedded_points) < n_neighbors:
        return 0

    # Step 2: Find nearest neighbors for each point
    divergences = []
    for i in range(len(embedded_points) - 1):
        # Compute distances to all other points
        distances = [euclidean(embedded_points[i], embedded_points[j]) for j in range(len(embedded_points)) if j != i]
        if len(distances) < n_neighbors:
            continue
        # Find indices of nearest neighbors
        nearest_indices = np.argsort(distances)[:n_neighbors]
        # Compute divergence after one time step
        for idx in nearest_indices:
            idx = idx if idx < i else idx + 1  # Adjust index after excluding i
            if i + 1 < len(embedded_points) and idx + 1 < len(embedded_points):
                divergence = euclidean(embedded_points[i + 1], embedded_points[idx + 1])
                if divergence > 0 and distances[idx] > 0:
                    log_div = np.log(divergence / distances[idx])
                    divergences.append(log_div)

    if not divergences:
        return 0

    # Step 3: Compute average divergence rate (LLE approximation)
    lle = np.mean(divergences)
    return lle if np.isfinite(lle) else 0


def dominant_freq_in_range(freqs, psd, fmin=0.5, fmax=5):
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0
    return freqs[mask][np.argmax(psd[mask])]


def calculate_circularity_index(disp_x, disp_y):
    """
    Calculate circularity index as the ratio of cumulative displacements in two axes.
    Returns the absolute ratio of disp_x to disp_y.
    """
    if disp_y == 0:
        return 0
    return abs(disp_x / disp_y)


def extract_features_per_axis(signal, sampling_rate):
    features = []
    centered = butter_lowpass_filter(signal - np.mean(signal), cutoff=5, fs=sampling_rate)
    freqs, psd = welch(centered, fs=sampling_rate, nperseg=16)  # Adjusted nperseg
    fmin, fmax = 0.5, 5

    mean = np.mean(centered)
    std = np.std(centered)
    rms = np.sqrt(np.mean(centered ** 2))
    peak_to_peak = np.max(centered) - np.min(centered)
    median = np.median(centered)
    min_val = np.min(centered)
    max_val = np.max(centered)
    signal_var = np.var(centered)
    zc_count = zero_crossings(centered)
    zc_var = zero_crossing_variance(centered)
    env_mean = calculate_envelope_stats(centered)
    slope_mean = calculate_slope(centered)
    trajectory = calculate_motion_trajectories(centered, sampling_rate)
    wl = calculate_waveform_length(centered)

    skewness = calculate_skewness(centered)
    kurt = calculate_kurtosis(centered)

    energy = np.sum(psd[(freqs >= 0.5) & (freqs <= 5)])
    entropy = calculate_spectral_entropy(centered, sampling_rate)
    dom_freq = dominant_freq_in_range(freqs, psd, fmin, fmax)
    thd = calculate_thd(signal, sampling_rate)
    second_ratio = calculate_harmonic_ratios(freqs, psd, fmin, fmax)
    low_freq_power = calculate_low_freq_power(psd, freqs, fmin=0, fmax=0.5)

    cycle_count_peaks = cycle_count(centered, sampling_rate)
    pos_to_neg = calculate_cpi(centered)
    num_transitions = calculate_directional_transitions(centered)

    # Combine all features (25 features per axis)
    features.extend([
        mean, std, rms, energy, skewness, kurt,
        peak_to_peak, median, min_val, max_val, signal_var,
        zc_count, zc_var, thd, entropy, dom_freq,
        env_mean, slope_mean, trajectory, wl,
        cycle_count_peaks, pos_to_neg, num_transitions,
        second_ratio, low_freq_power
    ])
    return features


def extract_features(data_path, X_matrix, features_to_extract):
    feature_list = []
    extract_all = features_to_extract == 'all'
    missing_data_flags = []

    # Define feature names based on the selected 186 features
    feature_names = []
    # For Acc and Gyro on magnitude only (8 features each)
    for sensor in ['acc', 'gyro']:
        feature_names.extend([
            f"{sensor}_magnitude_mean",
            f"{sensor}_magnitude_std",
            f"{sensor}_magnitude_rms",
            f"{sensor}_magnitude_energy",
            f"{sensor}_magnitude_skewness",
            f"{sensor}_magnitude_kurtosis",
            f"{sensor}_magnitude_spectral_entropy",
            f"{sensor}_magnitude_cycle_symmetry",
        ])
    # For Acc and Gyro on each axis (25 features per axis, 3 axes, 2 sensors)
    for sensor in ['acc', 'gyro']:
        for axis in ['x', 'y', 'z']:
            feature_names.extend([
                f"{sensor}_{axis}_mean",
                f"{sensor}_{axis}_std",
                f"{sensor}_{axis}_rms",
                f"{sensor}_{axis}_energy",
                f"{sensor}_{axis}_skewness",
                f"{sensor}_{axis}_kurtosis",
                f"{sensor}_{axis}_peak_to_peak",
                f"{sensor}_{axis}_median",
                f"{sensor}_{axis}_min",
                f"{sensor}_{axis}_max",
                f"{sensor}_{axis}_signal_var",
                f"{sensor}_{axis}_zero_crossings",
                f"{sensor}_{axis}_zero_crossing_variance",
                f"{sensor}_{axis}_thd",
                f"{sensor}_{axis}_spectral_entropy",
                f"{sensor}_{axis}_dominant_freq",
                f"{sensor}_{axis}_envelope_mean",
                f"{sensor}_{axis}_slope_mean",
                f"{sensor}_{axis}_trajectory",
                f"{sensor}_{axis}_wl",
                f"{sensor}_{axis}_cycle_count_peaks",
                f"{sensor}_{axis}_pos_to_neg",
                f"{sensor}_{axis}_num_transitions",
                f"{sensor}_{axis}_second_harmonic_ratio",
                f"{sensor}_{axis}_low_freq_power",
            ])
    # For Acc only on each axis (LLE, 1 feature per axis, 3 axes)
    for axis in ['x', 'y', 'z']:
        feature_names.append(f"acc_{axis}_lle")
    # Shared features for Acc and Gyro (6 features each)
    for sensor in ['acc', 'gyro']:
        feature_names.extend([
            f"{sensor}_xy_correlation",
            f"{sensor}_xz_correlation",
            f"{sensor}_yz_correlation",
            f"{sensor}_circularity_xy",
            f"{sensor}_circularity_xz",
            f"{sensor}_circularity_yz",
        ])
    # For Mag on magnitude (2 features)
    feature_names.extend([
        "mag_magnitude_mean",
        "mag_magnitude_std",
    ])
    # For Mag on each axis (1 feature per axis, 3 axes)
    for axis in ['x', 'y', 'z']:
        feature_names.append(f"mag_{axis}_signal_var")

    group_str = os.path.basename(data_path)
    data_cache = {}

    for recording_num in set(X_matrix[:, 4].astype(int)):
        for hand in [0, 1]:
            hand_str = 'L' if hand == 0 else 'R'
            for sensor in ['Acc', 'Gyro', 'Mag']:
                file_name = f"{group_str}_{recording_num:02d}_{hand_str}_{sensor}.csv"
                file_path = os.path.join(data_path, file_name)
                if os.path.isfile(file_path):
                    try:
                        data = pd.read_csv(file_path)
                        time_col = [col for col in data.columns if 'elapsed' in col][0]
                        x_col = [col for col in data.columns if 'x-axis' in col][0]
                        y_col = [col for col in data.columns if 'y-axis' in col][0]
                        z_col = [col for col in data.columns if 'z-axis' in col][0]
                        data_cache[file_path] = (data, time_col, x_col, y_col, z_col)
                    except Exception as e:
                        print(f"Error reading {file_name}: {str(e)}")
                        data_cache[file_path] = None

    total_windows = X_matrix.shape[0]

    for window_idx, window in enumerate(X_matrix):
        start_time, end_time, hand, group, recording = window
        recording_num = int(recording)
        hand_str = 'L' if hand == 0 else 'R'

        window_features = []
        sensor_types = ['Acc', 'Gyro', 'Mag']
        found_file = False
        has_missing_data = False

        for sensor in sensor_types:
            sampling_rate = 50 if sensor in ['Acc', 'Gyro'] else 25
            file_name = f"{group_str}_{recording_num:02d}_{hand_str}_{sensor}.csv"
            file_path = os.path.join(data_path, file_name)

            if file_path not in data_cache or data_cache[file_path] is None:
                if extract_all:
                    if sensor in ['Acc', 'Gyro']:
                        num_features = 8 + 25 * 3
                        if sensor == 'Acc':
                            num_features += 3
                        num_features += 6
                    else:
                        num_features = 2 + 3
                    window_features.extend([np.nan] * num_features)
                else:
                    window_features.extend([np.nan] * 3)
                found_file = False
                has_missing_data = True
                continue

            found_file = True
            data, time_col, x_col, y_col, z_col = data_cache[file_path]
            mask = (data[time_col] >= start_time) & (data[time_col] < end_time)
            window_data = data.loc[mask, [x_col, y_col, z_col]].values

            if window_data.shape[0] == 0:
                print(f"No data in window {start_time}-{end_time} for {file_name}")
                if extract_all:
                    if sensor in ['Acc', 'Gyro']:
                        num_features = 8 + 25 * 3
                        if sensor == 'Acc':
                            num_features += 3
                        num_features += 6
                    else:
                        num_features = 2 + 3
                    window_features.extend([np.nan] * num_features)
                else:
                    window_features.extend([np.nan] * 3)
                has_missing_data = True
                continue

            if np.any(np.isnan(window_data)) or np.any(np.isinf(window_data)):
                print(f"NaN/Inf in {file_name}, window {start_time}-{end_time}")
                for axis in range(window_data.shape[1]):
                    axis_data = window_data[:, axis]
                    valid_mask = np.isfinite(axis_data)
                    if np.any(valid_mask):
                        mean_value = np.mean(axis_data[valid_mask])
                        window_data[~valid_mask, axis] = mean_value
                    else:
                        window_data[~valid_mask, axis] = 0
                        print(f"All values invalid in axis {axis} for {file_name}")

            magnitude = np.sqrt(np.sum(window_data ** 2, axis=1))

            if extract_all:
                if sensor in ['Acc', 'Gyro']:
                    # Features on magnitude (8 features)
                    centered = butter_lowpass_filter(magnitude - np.mean(magnitude), cutoff=5, fs=sampling_rate)
                    freqs, psd = welch(centered, fs=sampling_rate, nperseg=16)
                    magnitude_features = [
                        np.mean(magnitude),  # Mean
                        np.std(magnitude),  # Std
                        np.sqrt(np.mean(magnitude ** 2)),  # RMS
                        np.sum(psd[(freqs >= 0.5) & (freqs <= 5)]),  # Energy
                        calculate_skewness(magnitude),  # Skewness
                        calculate_kurtosis(magnitude),  # Kurtosis
                        calculate_spectral_entropy(centered, sampling_rate),  # Spectral Entropy
                        calculate_cycle_consistency(magnitude, sampling_rate),  # Cycle Symmetry
                    ]

                    # Features on each axis (25 features per axis)
                    x_features = extract_features_per_axis(window_data[:, 0], sampling_rate)
                    y_features = extract_features_per_axis(window_data[:, 1], sampling_rate)
                    z_features = extract_features_per_axis(window_data[:, 2], sampling_rate)

                    # LLE for Acc only (3 features)
                    if sensor == 'Acc':
                        lle_features = [
                            calculate_largest_lyapunov_exponent(window_data[:, 0]),  # x_LLE
                            calculate_largest_lyapunov_exponent(window_data[:, 1]),  # y_LLE
                            calculate_largest_lyapunov_exponent(window_data[:, 2]),  # z_LLE
                        ]
                    else:
                        lle_features = []

                    # Shared features: Correlations and Circularity Indices (6 features)
                    corr_xy = np.corrcoef(window_data[:, 0], window_data[:, 1])[0, 1] if window_data.shape[0] > 1 else 0
                    corr_xz = np.corrcoef(window_data[:, 0], window_data[:, 2])[0, 1] if window_data.shape[0] > 1 else 0
                    corr_yz = np.corrcoef(window_data[:, 1], window_data[:, 2])[0, 1] if window_data.shape[0] > 1 else 0
                    traj_x = x_features[18]  # trajectory index in extract_features_per_axis
                    traj_y = y_features[18]
                    traj_z = z_features[18]
                    circ_xy = calculate_circularity_index(traj_x, traj_y)
                    circ_xz = calculate_circularity_index(traj_x, traj_z)
                    circ_yz = calculate_circularity_index(traj_y, traj_z)

                    window_features.extend(magnitude_features + x_features + y_features + z_features + lle_features +
                                           [corr_xy, corr_xz, corr_yz, circ_xy, circ_xz, circ_yz])
                else:
                    # Features on magnitude (2 features)
                    mag_features = [
                        np.mean(magnitude),  # Mean
                        np.std(magnitude),  # Std
                    ]
                    # Features on each axis (1 feature: Signal Variance)
                    axis_features = [
                        np.var(window_data[:, 0]),  # x_signal_var
                        np.var(window_data[:, 1]),  # y_signal_var
                        np.var(window_data[:, 2]),  # z_signal_var
                    ]
                    window_features.extend(mag_features + axis_features)
            else:
                try:
                    thd_x = calculate_thd(window_data[:, 0], sampling_rate)
                    thd_y = calculate_thd(window_data[:, 1], sampling_rate)
                    thd_z = calculate_thd(window_data[:, 2], sampling_rate)
                    window_features.extend([thd_x, thd_y, thd_z])
                except Exception as e:
                    print(f"Error in THD for {file_name}: {str(e)}")
                    window_features.extend([np.nan] * 3)

        feature_list.append(window_features)
        missing_data_flags.append(has_missing_data)

    X_features = np.array(feature_list)

    for col in range(X_features.shape[1]):
        invalid_mask = ~np.isfinite(X_features[:, col])
        if np.any(invalid_mask):
            feature_name = feature_names[col] if extract_all else \
                ['acc_thd_x', 'acc_thd_y', 'acc_thd_z', 'gyro_thd_x', 'gyro_thd_y', 'gyro_thd_z', 'mag_thd_x',
                 'mag_thd_y', 'mag_thd_z'][col]
            rows_with_invalid = np.where(invalid_mask)[0]
            all_missing_data = all(missing_data_flags[row] for row in rows_with_invalid)
            if all_missing_data:
                print(f"Replacing {np.sum(invalid_mask)} NaN/Inf in {feature_name} with 0")
                X_features[invalid_mask, col] = 0
            else:
                valid_mask = np.isfinite(X_features[:, col])
                if np.any(valid_mask):
                    valid_values = X_features[valid_mask, col]
                    mean_value = np.mean(valid_values)
                    min_value = np.min(valid_values)
                    max_value = np.max(valid_values)
                    if 'mean' in feature_name or 'std' in feature_name or 'variance' in feature_name:
                        replacement = mean_value
                    elif 'thd' in feature_name or 'entropy' in feature_name:
                        replacement = min_value
                    elif 'energy_ratio' in feature_name:
                        replacement = max_value
                    else:
                        replacement = mean_value
                    print(f"Replacing {np.sum(invalid_mask)} NaN/Inf in {feature_name} with {replacement}")
                    X_features[invalid_mask, col] = replacement
                else:
                    print(f"No valid values for {feature_name}. Using 0")
                    X_features[:, col] = 0

    if X_features.shape[0] != X_matrix.shape[0]:
        raise ValueError(f"Rows mismatch: X_features ({X_features.shape[0]}) vs X_matrix ({X_matrix.shape[0]})")

    return X_features


##################### Part C: Train & Test Split ######################################

def load_all_groups(temp_dirs, window_size, features_to_extract):
    """
    Load and combine data from all groups, ensuring proper concatenation of features across sensors.

    Args:
        temp_dirs (list): List of paths to temporary directories containing unzipped group folders.
        window_size (float): Window size for segmentation in seconds.
        features_to_extract (list or str): Features to extract (e.g., ['THD'] or 'all').

    Returns:
        X_matrix_all (np.array): Combined matrix of segmented windows from all groups.
        Y_vector_all (np.array): Combined vector of labels from all groups.
        X_features_all (np.array): Combined matrix of extracted features from all groups.
    """
    X_matrix_all = []
    Y_vector_all = []
    X_features_all = []

    for temp_dir_item in temp_dirs:
        group_dirs = [os.path.join(temp_dir_item, d) for d in os.listdir(temp_dir_item) if
                      os.path.isdir(os.path.join(temp_dir_item, d))]

        for group_dir in group_dirs:
            group_name = os.path.basename(group_dir)
            X_matrix, Y_vector = segment_signal(group_dir, window_size)

            # Track number of windows per recording and total for the group
            recording_nums = X_matrix[:, 4].astype(int)
            unique_recordings = np.unique(recording_nums)
            total_windows_group = 0
            for rec_num in unique_recordings:
                rec_windows = np.sum(recording_nums == rec_num)
                total_windows_group += rec_windows
            print(f"Total windows in group {group_name}: {total_windows_group}")

            X_features = extract_features(group_dir, X_matrix, features_to_extract)

            if X_features.shape[0] != X_matrix.shape[0]:
                print(
                    f"Error: Number of rows in X_features ({X_features.shape[0]}) does not match X_matrix ({X_matrix.shape[0]}) for group {group_dir}")
                raise ValueError("Mismatch in number of rows between X_features and X_matrix")

            X_matrix_all.append(X_matrix)
            Y_vector_all.append(Y_vector)
            X_features_all.append(X_features)

    if not X_matrix_all:
        print("No data found in any group directories.")
        return np.array([]), np.array([]), np.array([])

    X_matrix_all = np.vstack(X_matrix_all)
    Y_vector_all = np.hstack(Y_vector_all)

    if X_features_all and X_features_all[0].size > 0:
        feature_counts = [X.shape[1] for X in X_features_all]
        min_features = min(feature_counts)
        X_features_all = [X[:, :min_features] for X in X_features_all]
        X_features_all = np.vstack(X_features_all)
    else:
        X_features_all = np.array([])

    # Final check for row consistency across all groups
    if X_matrix_all.shape[0] != Y_vector_all.shape[0] or X_matrix_all.shape[0] != X_features_all.shape[0]:
        print(
            f"Error: Mismatch in number of rows - X_matrix_all: {X_matrix_all.shape[0]}, Y_vector_all: {Y_vector_all.shape[0]}, X_features_all: {X_features_all.shape[0]}")
        raise ValueError("Mismatch in number of rows between X_matrix_all, Y_vector_all, and X_features_all")

    return X_matrix_all, Y_vector_all, X_features_all


def split_data(X_features, Y_vector, X_matrix):
    """
    Split data into training and test sets using Split2 logic (80% groups train, 20% groups test).
    Remove undefined labels (-1) from the training set AFTER splitting to avoid data leakage,
    but keep them in the test set.

    Args:
        X_features (np.array): Feature matrix.
        Y_vector (np.array): Label vector.
        X_matrix (np.array): Metadata matrix.

    Returns:
        data_split (dict): Split2 results (X_train, X_test, Y_train, Y_test, indices, group IDs, test_original_indices).
    """
    data_split = {'X_train': [], 'X_test': [], 'Y_train': [], 'Y_test': [], 'train_idx': [], 'test_idx': [],
                  'train_group_ids': [], 'test_group_ids': [], 'test_original_indices': []}
    groups = np.unique(X_matrix[:, 3])

    # Split 2: 80% of groups for training, 20% for testing
    n_train_groups = int(0.8 * len(groups))
    train_groups = np.random.choice(groups, size=n_train_groups, replace=False)
    test_groups = np.setdiff1d(groups, train_groups)

    train_idx = np.isin(X_matrix[:, 3], train_groups)
    test_idx = np.isin(X_matrix[:, 3], test_groups)

    data_split['X_train'] = X_features[train_idx]
    data_split['X_test'] = X_features[test_idx]
    data_split['Y_train'] = Y_vector[train_idx]
    data_split['Y_test'] = Y_vector[test_idx]
    data_split['train_idx'] = np.where(train_idx)[0]
    data_split['test_idx'] = np.where(test_idx)[0]
    data_split['train_group_ids'] = X_matrix[train_idx, 3]
    data_split['test_group_ids'] = X_matrix[test_idx, 3]
    data_split['test_original_indices'] = np.where(test_idx)[0]

    # Remove undefined labels (-1) from training set
    if data_split['Y_train'].size > 0:
        train_valid_idx = data_split['Y_train'] != -1
        data_split['X_train'] = data_split['X_train'][train_valid_idx]
        data_split['Y_train'] = data_split['Y_train'][train_valid_idx]
        data_split['train_idx'] = data_split['train_idx'][train_valid_idx]
        data_split['train_group_ids'] = data_split['train_group_ids'][train_valid_idx]

    if data_split['Y_train'].size > 0 and data_split['Y_test'].size > 0:
        train_labels, train_counts = np.unique(data_split['Y_train'].astype(int), return_counts=True)
        test_labels, test_counts = np.unique(data_split['Y_test'].astype(int), return_counts=True)
        train_distribution = dict(zip(train_labels, train_counts))
        test_distribution = dict(zip(test_labels, test_counts))
        print("Split 2 - Training set label distribution (removing -1 from Train):", train_distribution)
        print("Split 2 - Test set label distribution (keeping -1 in Test):", test_distribution)
    else:
        print("Split 2 - No data in training or testing sets after processing.")

    return data_split


##################### Part D: Discretization Function (Unified) ##########################

def discretize_with_constant_frequency(data, labels, num_bins=50):
    """
    Discretizes a 1D array using the Constant Frequency method with a specified number of bins.
    Each bin contains an equal number of samples.

    Args:
        data (np.array): 1D array of numerical values to discretize.
        labels (np.array): 1D array of categorical labels (not used in Constant Frequency, kept for compatibility).
        num_bins (int): Number of bins for discretization (default=100).

    Returns:
        np.array: Array of bin indices (categories) for each value.
    """
    # Sort the data and keep track of indices
    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]

    # Calculate the number of samples per bin
    n_samples = len(data)
    samples_per_bin = n_samples // num_bins
    remainder = n_samples % num_bins

    # Assign bin indices
    bin_assignments = np.zeros(n_samples, dtype=int)
    current_idx = 0
    for bin_idx in range(num_bins):
        # Adjust the number of samples in this bin to account for the remainder
        bin_size = samples_per_bin + (1 if bin_idx < remainder else 0)
        if bin_size == 0:
            continue
        # Assign the bin index to the samples in this range
        bin_assignments[current_idx:current_idx + bin_size] = bin_idx
        current_idx += bin_size

    # Map back to original order
    result = np.zeros_like(data, dtype=int)
    result[sorted_indices] = bin_assignments

    return result


##################### Part E: Feature Correlation Analysis ##########################

def THD_EXTRACTION(temp_dir):
    """
    Extract THD features for different window sizes (1, 3, 10) and combine them into a single matrix.
    Processes all groups within the temporary directory. Updated to handle THD features for all sensors (Acc, Gyro, Mag).
    Applies Constant Frequency discretization to the THD features.

    Args:
        temp_dir (str): Path to temporary directory containing unzipped group folders.

    Returns:
        X_features_THD (np.array): Combined THD features matrix (9 columns: 3 axes × 3 sensors), discretized into categories.
        Y_vectors_THD (list): List of label vectors.
        window_size_indices (list): List of tuples (start_idx, end_idx, window_size).
    """
    window_sizes = [1, 3, 10]
    X_features_THD_list = []
    Y_vectors_THD = []
    window_size_indices = []
    current_idx = 0

    thd_feature_names = [
        'acc_thd_x', 'acc_thd_y', 'acc_thd_z',
        'gyro_thd_x', 'gyro_thd_y', 'gyro_thd_z',
        'mag_thd_x', 'mag_thd_y', 'mag_thd_z'
    ]

    for window_size in window_sizes:
        print(f"\nProcessing window size {window_size}...")
        X_matrix_all, Y_vector_all, X_features_all = load_all_groups([temp_dir], window_size,
                                                                     features_to_extract=['THD'])

        if X_features_all.size > 0 and X_features_all.shape[1] != 9:
            print(f"Error: Expected 9 THD features for window_size {window_size}, but got {X_features_all.shape[1]}")
            continue

        # Apply Constant Frequency discretization to each feature (column) in X_features_all
        discretized_features = np.zeros_like(X_features_all, dtype=int)
        for col_idx in range(X_features_all.shape[1]):
            original_data = X_features_all[:, col_idx]
            categories = discretize_with_constant_frequency(original_data, Y_vector_all, num_bins=50)

            discretized_features[:, col_idx] = categories

        X_features_THD_list.append(discretized_features)
        Y_vectors_THD.append(Y_vector_all)

        start_idx = current_idx
        end_idx = current_idx + (discretized_features.shape[0] if discretized_features.size > 0 else 0)
        window_size_indices.append((start_idx, end_idx, window_size))
        current_idx = end_idx

    if not X_features_THD_list or not any(X_features_all.size > 0 for X_features_all in X_features_THD_list):
        print("Warning: No THD features extracted for any window size.")  # Added warning
        return np.array([]), Y_vectors_THD, window_size_indices

    # Ensure X_features_THD is an empty array if X_features_THD_list is empty
    if not X_features_THD_list:
        X_features_THD = np.array([])
    else:
        X_features_THD = np.vstack(X_features_THD_list)

    return X_features_THD, Y_vectors_THD, window_size_indices


def feature_correlation(X_features_THD, Y_vectors_THD, window_size_indices):
    """
    Compute Mutual Information between discretized THD features (categories) and labels for different window sizes
    to determine the best window size. Forces the final window size to be 3, even if another size is selected.

    Args:
        X_features_THD (np.array): Combined THD features matrix (categories).
        Y_vectors_THD (list): List of label vectors.
        window_size_indices (list): List of tuples (start_idx, end_idx, window_size).

    Returns:
        float: Best window size (forced to 3).
    """
    window_mi = {}

    for start_idx, end_idx, window_size in window_size_indices:
        print(f"\nAnalyzing Mutual Information for window size {window_size}")
        thd_features = X_features_THD[start_idx:end_idx]

        y_vector_idx = window_size_indices.index((start_idx, end_idx, window_size))
        Y_vector = Y_vectors_THD[y_vector_idx]

        valid_idx = Y_vector != -1
        thd_features = thd_features[valid_idx]
        Y_vector_valid = Y_vector[valid_idx]

        if thd_features.shape[0] == 0:
            print(f"No valid data for window size {window_size} after removing undefined labels")
            window_mi[window_size] = 0
            continue

        mi_scores = []
        for thd_idx in range(thd_features.shape[1]):
            thd_labels = thd_features[:, thd_idx]
            if np.any(~np.isfinite(thd_labels)):
                print(
                    f"Warning: Unexpected NaN/Inf values in thd_features for window size {window_size}, feature index {thd_idx}. Skipping MI calculation for this feature.")
                mi_scores.append(0)
                continue
            mi = mutual_info_score(thd_labels, Y_vector_valid)
            mi_scores.append(mi)

        avg_mi = np.mean(mi_scores)
        window_mi[window_size] = avg_mi
        print(f"Window size {window_size}: Average Mutual Information with labels = {avg_mi:.4f}")

    if not window_mi:
        print("No window sizes were processed successfully. Defaulting to window size 3.")
        return 3

    best_window = max(window_mi, key=window_mi.get)
    print(f"\nBest window size: {best_window} with average Mutual Information {window_mi[best_window]:.4f}")

    if best_window != 3:
        print(f"The best window size selected was {best_window}, but we proceeded with a window size of 3")
        best_window = 3

    return best_window


############################## Part F: Vetting & Normalization #########################################

def _apply_preprocessing_steps(X_data, Y_data, group_ids, split_type, normalization_params=None, apply_fit=True,
                               feature_names=None):
    """
    Applies NaN/Inf handling, outlier capping, and normalization to the data.
    This version applies GLOBAL normalization (suitable for Split2).

    Args:
        X_data (np.ndarray): The feature matrix to preprocess.
        Y_data (np.ndarray): The corresponding labels (used for MI, though not directly in all steps here).
        group_ids (np.ndarray): Group IDs for samples (not used for normalization in this global approach, but kept for compatibility).
        split_type (str): 'training', 'test_internal', or 'finaltest' for logging purposes.
        normalization_params (dict, optional): Pre-learned normalization parameters (means, stds).
                                              If None and apply_fit is True, parameters are learned.
        apply_fit (bool): If True, learn normalization parameters. If False, use provided normalization_params.
        feature_names (list, optional): List of feature names for warnings.

    Returns:
        X_preprocessed (np.ndarray): The preprocessed feature matrix.
        learned_norm_params (dict): The learned normalization parameters (only if apply_fit is True).
    """
    # --- Step 0: Handle NaN/Inf values ---
    # Note: More comprehensive NaN/Inf handling is performed during the initial feature extraction,
    # this step serves as an additional safeguard
    X_cleaned = X_data.copy()
    if np.any(np.isnan(X_cleaned)) or np.any(np.isinf(X_cleaned)):
        print(f"Warning: Found NaN/Inf values in data for {split_type}. Handling...")
        for col in range(X_cleaned.shape[1]):
            column = X_cleaned[:, col]
            valid_mask = np.isfinite(column)
            if np.any(valid_mask):
                mean_value = np.mean(column[valid_mask])
                X_cleaned[~valid_mask, col] = mean_value
            else:
                X_cleaned[~valid_mask, col] = 0
                if feature_names:
                    print(f"Warning: No valid values in column {feature_names[col]}. Using 0 as fallback.")

    # --- Step 1: Outlier Detection and Handling using IQR method ---
    print(f"Handling outliers in data for {split_type} using IQR method...")
    for col in range(X_cleaned.shape[1]):
        feature = X_cleaned[:, col]
        Q1 = np.percentile(feature, 25)
        Q3 = np.percentile(feature, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        X_cleaned[feature < lower_bound, col] = lower_bound
        X_cleaned[feature > upper_bound, col] = upper_bound

    # --- Step 2: Normalization (Always Global for Split2 logic) ---
    X_normalized = X_cleaned.copy()
    current_norm_params = {}

    if apply_fit:
        # Learn global normalization parameters from the entire X_cleaned data
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X_cleaned)
        current_norm_params['all'] = {'mean': scaler.mean_.tolist(),
                                      'std': scaler.scale_.tolist()}  # Convert to list for JSON
        print(f"Learned GLOBAL normalization parameters for {split_type}.")
    else:  # Apply normalization based on provided global parameters
        if normalization_params is None or 'all' not in normalization_params:
            raise ValueError("Global normalization parameters ('all' key) must be provided when apply_fit is False.")

        means = np.array(normalization_params['all']['mean'])  # Convert back to numpy array
        stds = np.array(normalization_params['all']['std'])  # Convert back to numpy array
        stds[stds == 0] = 1e-8  # Avoid division by zero
        X_normalized = (X_cleaned - means) / (stds)
        print(f"Applied GLOBAL normalization using learned parameters for {split_type}.")

    return X_normalized, current_norm_params if apply_fit else None


def vet_features(X_preprocessed_data, Y_train, train_group_ids, split_type, feature_names):
    """
    Vets (selects relevant and non-redundant) features.
    Assumes X_preprocessed_data is already preprocessed (NaN/Inf, outliers, normalized).

    Args:
        X_preprocessed_data (np.array): Training feature matrix, ALREADY PREPROCESSED.
        Y_train (np.array): Training labels.
        train_group_ids (np.array): Group IDs for training samples (not directly used in this function, but kept for compatibility).
        split_type (str): 'split2' (split_type is only used for logging here).
        feature_names (list): Full list of 186 feature names.

    Returns:
        X_vetting (np.array): Filtered and selected training feature matrix (after MI/ReliefF).
        selected_feature_names (list): Names of the selected features after vetting.
    """
    # X_preprocessed_data is already cleaned and normalized by _apply_preprocessing_steps

    # Step 3: Compute MI between each feature and the label
    mi_scores = []
    for col_idx in range(X_preprocessed_data.shape[1]):
        feature = X_preprocessed_data[:, col_idx]
        feature_labels = discretize_with_constant_frequency(feature, Y_train, num_bins=100)
        mi = mutual_info_score(feature_labels, Y_train)
        mi_scores.append(mi)
        # print(f"Feature {feature_names[col_idx]}: MI with label = {mi:.4f}")

    mi_scores = np.array(mi_scores)

    # Step 4: Remove highly correlated features (Spearman > 0.8), keeping the feature with higher MI
    df_preprocessed = pd.DataFrame(X_preprocessed_data)
    corr_matrix = df_preprocessed.corr(method='spearman').to_numpy()
    corr_matrix = np.abs(corr_matrix)
    to_remove = set()
    for i in range(corr_matrix.shape[0]):
        if i in to_remove:
            continue
        for j in range(i + 1, corr_matrix.shape[1]):
            if j in to_remove:
                continue
            if corr_matrix[i, j] > 0.8:
                if mi_scores[i] > mi_scores[j]:
                    to_remove.add(j)
                    print(
                        f"Removing feature {feature_names[j]} (MI={mi_scores[j]:.4f}) in favor of {feature_names[i]} (MI={mi_scores[i]:.4f}) due to high correlation ({corr_matrix[i, j]:.4f})")
                else:
                    to_remove.add(i)
                    print(
                        f"Removing feature {feature_names[i]} (MI={mi_scores[i]:.4f}) in favor of {feature_names[j]} (MI={mi_scores[j]:.4f}) due to high correlation ({corr_matrix[i, j]:.4f})")
                    break

    keep_indices = [i for i in range(X_preprocessed_data.shape[1]) if i not in to_remove]
    X_train_filtered = X_preprocessed_data[:, keep_indices]
    filtered_feature_names = [feature_names[i] for i in keep_indices]

    print(
        f"Removed {len(to_remove)} features with correlation > 0.8 for {split_type}. Remaining features: {len(keep_indices)}")

    # Step 5: Select top 20 features using ReliefF
    relief = ReliefF(n_features_to_select=20, n_neighbors=100)
    relief.fit(X_train_filtered, Y_train)
    selected_indices = relief.top_features_[:20]
    selected_indices = selected_indices[:min(20, len(filtered_feature_names))]
    X_vetting = X_train_filtered[:, selected_indices]
    selected_feature_names = [filtered_feature_names[i] for i in selected_indices]

    print(f"After ReliefF selection, {X_vetting.shape[1]} features remain")
    print(f"Selected top {X_vetting.shape[1]} features: {selected_feature_names}")

    return X_vetting, selected_feature_names


############################## Part G: MRMR Feature Selection #########################################


def select_features_mrmr(X, y, split_name, feature_names, max_features=10, miq_threshold=0.2):
    """
    Selects features using MRMR (Max-Relevance Min-Redundancy) as described in the lecture.
    Use categorical values directly from Constant Frequency discretization.

    Args:
        X (np.ndarray): Input features (already normalized from vet_features).
        y (np.ndarray): Target variable.
        split_name (str): Name of the split ('split2').
        feature_names (list): List of feature names.
        max_features (int): Maximum number of features to select.
        miq_threshold (float): Minimum MIQ value to continue selecting features.

    Returns:
        tuple: (X_switched, selected_indices)
    """
    print(f"Starting MRMR feature selection for {split_name}")

    # Ensure y is 1D
    y = y.ravel()

    # Discretize features using Constant Frequency (since X is normalized but not discretized)
    n_samples, n_features = X.shape
    X_discretized = np.zeros_like(X, dtype=int)
    valid_features = []

    # Apply Constant Frequency discretization to each feature (returns categories directly)
    for feature_idx in range(n_features):
        feature = X[:, feature_idx].copy()
        if len(np.unique(feature)) < 2:
            X_discretized[:, feature_idx] = feature.astype(int)
        else:
            discretized_feature = discretize_with_constant_frequency(feature, y,
                                                                     num_bins=np.min([100, len(np.unique(feature))]))
            X_discretized[:, feature_idx] = discretized_feature
            valid_features.append(feature_idx)

    # Start MRMR
    n_features = X_discretized.shape[1]
    selected = []
    remaining = list(range(n_features))

    # Filter remaining features to only include valid ones (with enough unique values)
    remaining = [i for i in remaining if i in valid_features]
    if not remaining:
        print("No valid features with enough unique values for MRMR. Selecting default features.")
        selected_indices = [0, 1, 2, 3][:min(4, X.shape[1])]
        X_switched = X[:, selected_indices]
        return X_switched, selected_indices

    # Step 1: Select the feature with the largest relevance, ensuring MI > 0
    mi_scores = mutual_info_classif(X_discretized[:, remaining], y.ravel(), random_state=42)

    # Map MI scores to the corresponding feature indices in remaining
    mi_scores_dict = {remaining[i]: mi_scores[i] for i in range(len(remaining))}
    # Filter out features with MI = 0
    filtered_remaining = [f for f in remaining if mi_scores_dict[f] > 0]

    if not filtered_remaining:
        print("No features with MI > 0 for MRMR. Selecting default features.")
        selected_indices = [0, 1, 2, 3][:min(4, X.shape[1])]
        X_switched = X[:, selected_indices]
        return X_switched, selected_indices

    # Select the first feature
    best_idx_in_filtered = np.argmax([mi_scores_dict[f] for f in filtered_remaining])
    best_feature = filtered_remaining[best_idx_in_filtered]
    selected.append(best_feature)
    filtered_remaining.pop(best_idx_in_filtered)
    print(
        f"Step 1: Selected feature {feature_names[best_feature]} (original index {best_feature}) with MI score {mi_scores_dict[best_feature]:.4f}")

    # Step 2: Iteratively select features based on MIQ
    while filtered_remaining and len(selected) < max_features:
        best_miq = float('-inf')
        best_feature = None
        best_relevance = 0  # Re-introducing these variables for previous print logic
        best_redundancy = 0 # Re-introducing these variables for previous print logic

        # Pre-compute MI to selected features for efficiency
        mi_to_selected = {}
        for sel_f in selected:
            for rem_f in filtered_remaining:
                if (sel_f, rem_f) not in mi_to_selected and (rem_f, sel_f) not in mi_to_selected:
                    mi_val = mutual_info_score(X_discretized[:, sel_f], X_discretized[:, rem_f])
                    mi_to_selected[(sel_f, rem_f)] = mi_val
                    mi_to_selected[(rem_f, sel_f)] = mi_val

        for feature in filtered_remaining:
            # Compute relevance
            relevance_candidate = mutual_info_classif(X_discretized[:, [feature]], y.ravel(), random_state=42)[0]

            # Compute redundancy with already selected features
            redundancy_sum = 0
            for sel_f in selected:
                redundancy_sum += mi_to_selected.get((sel_f, feature), 0)
            redundancy_candidate = redundancy_sum / len(selected) if len(selected) > 0 else 1

            if redundancy_candidate > 0:
                miq = relevance_candidate / redundancy_candidate
                if miq > best_miq and relevance_candidate > 0:
                    best_miq = miq
                    best_feature = feature
                    best_relevance = relevance_candidate # Keep track for print in previous version
                    best_redundancy = redundancy_candidate # Keep track for print in previous version

        # Stop if no feature with positive relevance and redundancy is found
        if best_feature is None:
            print(f"No more features with positive relevance and redundancy for selection. Stopping.")
            break

        # Stop if the best MIQ is below the threshold
        if best_miq < miq_threshold:
            print(f"Stopping because MIQ ({best_miq:.4f}) is below threshold ({miq_threshold}).")
            break

        selected.append(best_feature)
        filtered_remaining.remove(best_feature)
        # Re-introducing the more verbose print from the older version
        print(f"Selected feature {feature_names[best_feature]} (original index {best_feature}) with MIQ {best_miq:.4f} (relevance {best_relevance:.4f} / redundancy {best_redundancy:.4f})")

    print(
        f"Selected {len(selected)} features for {split_name} (original indices: {selected}): {[feature_names[i] for i in selected]}")
    X_switched = X[:, selected]
    selected_indices_to_save = [int(idx) for idx in selected]
    return X_switched, selected_indices_to_save

############################## Part H: Model Training with Grid Search #########################################


def train_model(X_switched, Y_train):
    """
    Trains SVM model using Stratified K-Fold cross-validation.
    Performs Grid Search with 5-fold CV to select the best hyperparameters.
    Computes accuracy on the training set using cross-validation.

    Args:
        X_switched (np.ndarray): Selected features after MRMR.
        Y_train (np.ndarray): Training labels.

    Returns:
        model_a (estimator): Trained SVM model.
    """
    classes = {0: 'Rest', 1: 'Up-Down', 2: 'Right-Left', 3: 'Circular'}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Grid Search for SVM
    svm_param_grid = {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['linear', 'rbf']
    }
    base_svm = SVC(
        gamma='scale',
        probability=True,
        random_state=42
    )
    svm_grid = GridSearchCV(
        estimator=base_svm,
        param_grid=svm_param_grid,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1
    )
    svm_grid.fit(X_switched, Y_train)
    print(f"Best SVM parameters: {svm_grid.best_params_}")

    # Evaluate SVM with best parameters using 5-fold CV
    svm_scores = []
    for train_idx, val_idx in skf.split(X_switched, Y_train):
        X_train_fold, X_val_fold = X_switched[train_idx], X_switched[val_idx]
        Y_train_fold, Y_val_fold = Y_train[train_idx], Y_train[val_idx]
        svm_fold_model = SVC(
            C=svm_grid.best_params_['C'],
            kernel=svm_grid.best_params_['kernel'],
            gamma='scale',
            probability=True,
            random_state=42
        )
        svm_fold_model.fit(X_train_fold, Y_train_fold)
        Y_pred = svm_fold_model.predict(X_val_fold)
        score = accuracy_score(Y_val_fold, Y_pred)
        svm_scores.append(score)
    print(f"SVM Cross-Validation Accuracy (Train): {np.mean(svm_scores):.4f} ± {np.std(svm_scores):.4f}")

    # Train final SVM model with best parameters
    model_a = SVC(
        C=svm_grid.best_params_['C'],
        kernel=svm_grid.best_params_['kernel'],
        gamma='scale',
        probability=True,
        random_state=42
    )
    model_a.fit(X_switched, Y_train)

    return model_a


############################## Part I: Model Evaluation #########################################

def evaluate_model(model_a, X_test, Y_test, split_name, test_group_ids, norm_params, selected_indices,
                   all_figures, X_matrix_full, test_original_indices, output_dir):
    """
    Evaluates the trained SVM model on the test set and generates ROC curves.
    This version computes ROC, CM, Sensitivity based on WINDOWS (like main_05_TEST_seed.py),
    while overall accuracy is also computed based on WINDOWS, INCLUDING -1 LABELS.

    Args:
        model_a (estimator): Trained SVM model.
        X_test (np.ndarray): Test features (window-based, preprocessed & feature-selected).
        Y_test (np.ndarray): Test labels (window-based).
        split_name (str): Name of the split ('split2').
        test_group_ids (np.ndarray): Group IDs for test samples (window-based).
        norm_params (dict): Normalization parameters.
        selected_indices (list): Indices of features selected after MRMR.
        all_figures (list): List to store matplotlib Figure objects for ROC curves.
        X_matrix_full (np.ndarray): The full X_matrix_all containing original window metadata.
        test_original_indices (np.ndarray): Original indices of test samples within X_matrix_full.
        output_dir (str): Directory to save output files.

    Returns:
        tuple: (roc_auc_a, sensitivity_a, cm_a, overall_accuracy_a)
    """
    # X_test received here is already preprocessed and feature-selected.
    # So, X_test_valid is simply X_test.
    X_test_valid = X_test
    Y_test_valid = Y_test  # Y_test is passed directly, still contains -1 labels.

    # Compute probabilities and predictions for the model
    if X_test_valid.shape[0] == 0 or X_test_valid.shape[1] == 0:
        print(f"Warning: X_test_valid is empty or has no features for {split_name}. Cannot perform predictions.")
        return [0] * 4, 0, np.zeros((4, 4)), 0

    Y_prob_a = model_a.predict_proba(X_test_valid)
    Y_pred_a_window = model_a.predict(X_test_valid)

    # overall accuracy now considers all labels including -1
    # The models are trained on 0,1,2,3 so they will predict one of these.
    # If Y_test_valid is -1, and Y_pred is not -1, it's a "misclassification" in this context.

    correct_predictions_a = (Y_pred_a_window == Y_test_valid).sum()
    overall_accuracy_a = correct_predictions_a / Y_test_valid.shape[0] if Y_test_valid.shape[0] > 0 else 0

    # Save expected accuracy to CSV
    expected_accuracy_path = os.path.join(output_dir, f"{GROUP_NUMBER}_expected_accuracy.csv")
    pd.DataFrame([overall_accuracy_a], columns=["Expected_Accuracy"]).to_csv(expected_accuracy_path, index=False)
    print(f"Expected accuracy saved to {expected_accuracy_path}")

    # ROC/CM/Sensitivity calculations will also use data, EXCLUDING -1 LABELS ---
    valid_idx_window_based = np.isin(Y_test_valid, [0, 1, 2, 3])

    Y_true_window_filtered = Y_test_valid[valid_idx_window_based]
    Y_pred_a_window_filtered = Y_pred_a_window[valid_idx_window_based]
    Y_prob_a_window_filtered = Y_prob_a[valid_idx_window_based]

    # Remaining calculations for ROC/CM/Sensitivity use the filtered data
    if Y_true_window_filtered.shape[0] == 0:
        print(
            f"Warning: No valid labeled samples found in test set for {split_name} after filtering -1 (window-based). Cannot compute ROC/CM/Sensitivity.")
        return [0] * 4, 0, np.zeros((4, 4)), overall_accuracy_a

    # Check if enough classes remain for multi_class AUC
    unique_labels_filtered = np.unique(Y_true_window_filtered)
    print(f"Unique labels found in filtered test set: {unique_labels_filtered}")  # Added print for unique labels
    if len(unique_labels_filtered) < 2:
        print(f"Warning: Only one class present in filtered test labels for {split_name}. Cannot compute AUC.")
        roc_auc_a = [0] * 4
    else:
        # Calculate ROC AUC scores using window-based probabilities and labels
        model_classes_a = model_a.classes_ if hasattr(model_a, 'classes_') else np.array([])

        prob_a_reordered_window = np.zeros((Y_prob_a_window_filtered.shape[0], 4))

        for i_cls, cls_val in enumerate(model_classes_a):
            if cls_val in [0, 1, 2, 3]:
                prob_a_reordered_window[:, int(cls_val)] = Y_prob_a_window_filtered[:, i_cls]

        try:
            # Ensure Y_true_window_filtered has unique values for all labels in labels parameter
            # labels parameter should only contain classes actually present in Y_true_window_filtered
            labels_for_auc = sorted(list(unique_labels_filtered.astype(int)))
            roc_auc_a = roc_auc_score(Y_true_window_filtered, prob_a_reordered_window[:, labels_for_auc],
                                      multi_class='ovr', average=None, labels=labels_for_auc).tolist()
            # Pad with zeros if some classes were not present in filtered test data
            full_roc_auc_a = [0] * 4
            for i, label in enumerate(labels_for_auc):
                full_roc_auc_a[label] = roc_auc_a[i]
            roc_auc_a = full_roc_auc_a
        except ValueError as e:
            print(
                f"Error computing ROC-AUC for {split_name}: {e}. This often indicates missing classes in predictions or true labels, or issue with probability array shape.")
            roc_auc_a = [0] * 4

    # Compute ROC curves for each class and each model using filtered WINDOW-BASED samples
    classes_map = {0: 'Rest', 1: 'Up-Down', 2: 'Right-Left', 3: 'Circular'}
    roc_data_a = []
    auc_per_class_a = []

    for cls in range(4):
        Y_binary = (Y_true_window_filtered == cls).astype(int)

        if len(np.unique(Y_binary)) < 2:
            roc_data_a.append(([], []))
            auc_per_class_a.append(0)
            continue

        fpr_a, tpr_a, _ = roc_curve(Y_binary, prob_a_reordered_window[:, cls])
        auc_a = roc_auc_score(Y_binary, prob_a_reordered_window[:, cls])
        roc_data_a.append((fpr_a, tpr_a))
        auc_per_class_a.append(auc_a)

    # Compute confusion matrices with filtered WINDOW-BASED predictions
    cm_a = confusion_matrix(Y_true_window_filtered, Y_pred_a_window_filtered, labels=[0, 1, 2, 3])

    # Compute sensitivity for Class 1 (Up-Down) using filtered WINDOW-BASED predictions
    tp_a = cm_a[1, 1] if 1 in unique_labels_filtered else 0
    fn_a = np.sum(cm_a[1, :]) - tp_a if 1 in unique_labels_filtered else 0
    sensitivity_a = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0

    # Display evaluation metrics
    print(f"\nEvaluation for {split_name}:")
    print(f"Model A (SVM) ROC-AUC per class: {[f'{auc:.4f}' for auc in roc_auc_a]}")
    print(f"Model A (SVM) Sensitivity for Class 1 (Up-Down): {sensitivity_a:.4f}")
    print(f"Model A (SVM) Confusion Matrix:\n{cm_a}")
    print(
        f"Model A (SVM) Overall Accuracy (Window-based, including -1 in total samples): {overall_accuracy_a:.4f}")

    # Create all figures for the current split and add them to the global list
    print(f"Generating ROC plots for {split_name}...")
    plot_count = 0
    for cls in range(4):
        fpr_a, tpr_a = roc_data_a[cls]
        auc_a = auc_per_class_a[cls]
        if len(fpr_a) > 0 and len(tpr_a) > 0:
            fig_a = plt.figure(figsize=(8, 6))
            plt.plot(fpr_a, tpr_a, label=f'ROC curve (AUC = {auc_a:.2f})', color='blue')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Chance')
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.title(
                f'ROC Curve - Model A (SVM) - {split_name} - Class {classes_map[cls]} (AUC = {auc_a:.2f})')
            plt.legend(loc="lower right")
            all_figures.append(fig_a)
            plot_count += 1

    print(f"Total ROC plots generated for {split_name}: {plot_count}")

    return roc_auc_a, sensitivity_a, cm_a, overall_accuracy_a


def export_predictions_csv(X_matrix, predictions, output_filename):
    """
    Exports predictions to a CSV file in the specified format: Start, End, Label.

    Args:
        X_matrix (np.ndarray): Original X_matrix containing start_time, end_time.
        predictions (np.ndarray): Array of predicted labels.
        output_filename (str): Name of the output CSV file.
    """
    if X_matrix.shape[0] != len(predictions):
        raise ValueError("Mismatch between number of windows in X_matrix and predictions.")

    df_predictions = pd.DataFrame({
        'Start': X_matrix[:, 0].astype(int),  # Convert to whole seconds
        'End': X_matrix[:, 1].astype(int),  # Convert to whole seconds
        'Label': predictions.astype(int)
    })
    df_predictions.to_csv(output_filename, index=False)
    print(f"Predictions saved to {output_filename}")


############################## Main: Combined Execution Pipeline #########################################

def main():
    """
    Main function to process data, extract features, split data, vet features, select features with MRMR,
    train models, and evaluate them, using Split2 logic exclusively.
    """
    output_dir = os.path.dirname(__file__)
    intermediate_output_dir = os.path.join(output_dir, 'data', 'intermediate_outputs')
    if not os.path.exists(intermediate_output_dir):
        os.makedirs(intermediate_output_dir)

    # Step 1: Initialize temporary directory for pool_dataset and extract data
    temp_dir_pool = tempfile.mkdtemp()
    zip_path_pool = os.path.join(output_dir, 'data', 'pool_dataset.zip')

    # Save the temporary directory path for later cleanup in Part 2
    with open(os.path.join(intermediate_output_dir, 'temp_dir_pool_path.txt'), 'w') as f:
        f.write(temp_dir_pool)

    if not os.path.exists(zip_path_pool):
        print(f"Error: ZIP file not found at path: {zip_path_pool}")
        print(
            "Please ensure 'pool_dataset.zip' is inside a folder named 'data' in the same directory as the Python file.")
        os.rmdir(temp_dir_pool)
        return

    print(f"Extracting pool_dataset.zip to {temp_dir_pool}")
    with zipfile.ZipFile(zip_path_pool, 'r') as zip_ref:
        zip_ref.extractall(temp_dir_pool)

    # Step 2: THD Extraction and Feature Correlation to determine best window size (forced to 3)
    X_features_THD, Y_vectors_THD, window_size_indices = THD_EXTRACTION(temp_dir_pool)
    best_window_size = feature_correlation(X_features_THD, Y_vectors_THD, window_size_indices)

    # Save best_window_size
    with open(os.path.join(intermediate_output_dir, 'best_window_size.txt'), 'w') as f:
        f.write(str(best_window_size))

    # Step 3: Load all groups with the best window size and extract all features
    print(f"\nLoading all groups from pool_dataset with window size {best_window_size} and extracting all features...")
    X_matrix_all, Y_vector_all, X_features_all = load_all_groups([temp_dir_pool], window_size=3,
                                                                 features_to_extract='all')

    # Save X_matrix_all (for reference)
    save_numpy_to_csv(X_matrix_all, 'X_matrix_all.csv', intermediate_output_dir)
    # Save Y_vector_all (for reference)
    save_numpy_to_csv(Y_vector_all, 'Y_vector_all.csv', intermediate_output_dir)
    # Save X_features_all (for reference)
    save_numpy_to_csv(X_features_all, 'X_features_all.csv', intermediate_output_dir)

    # Verify that the number of rows matches
    if X_matrix_all.shape[0] != Y_vector_all.shape[0] or X_matrix_all.shape[0] != X_features_all.shape[0]:
        print(
            f"Error: Mismatch in number of rows - X_matrix_all: {X_matrix_all.shape[0]}, Y_vector_all: {Y_vector_all.shape[0]}, X_features_all: {X_features_all.shape[0]}")
        raise ValueError("Mismatch in number of rows between X_matrix_all, Y_vector_all, and X_features_all")

    # Define all 186 feature names
    feature_names = [
                        f"{sensor}_magnitude_mean" for sensor in ['acc', 'gyro']
                    ] + [
                        f"{sensor}_magnitude_std" for sensor in ['acc', 'gyro']
                    ] + [
                        f"{sensor}_magnitude_rms" for sensor in ['acc', 'gyro']
                    ] + [
                        f"{sensor}_magnitude_energy" for sensor in ['acc', 'gyro']
                    ] + [
                        f"{sensor}_magnitude_skewness" for sensor in ['acc', 'gyro']
                    ] + [
                        f"{sensor}_magnitude_kurtosis" for sensor in ['acc', 'gyro']
                    ] + [
                        f"{sensor}_magnitude_spectral_entropy" for sensor in ['acc', 'gyro']
                    ] + [
                        f"{sensor}_magnitude_cycle_symmetry" for sensor in ['acc', 'gyro']
                    ] + [
                        f"{sensor}_{axis}_mean" for sensor in ['acc', 'gyro'] for axis in ['x', 'y', 'z']
                    ] + [
                        f"{sensor}_{axis}_std" for sensor in ['acc', 'gyro'] for axis in ['x', 'y', 'z']
                    ] + [
                        f"{sensor}_{axis}_rms" for sensor in ['acc', 'gyro'] for axis in ['x', 'y', 'z']
                    ] + [
                        f"{sensor}_{axis}_energy" for sensor in ['acc', 'gyro'] for axis in ['x', 'y', 'z']
                    ] + [
                        f"{sensor}_{axis}_skewness" for sensor in ['acc', 'gyro'] for axis in ['x', 'y', 'z']
                    ] + [
                        f"{sensor}_{axis}_kurtosis" for sensor in ['acc', 'gyro'] for axis in ['x', 'y', 'z']
                    ] + [
                        f"{sensor}_{axis}_peak_to_peak" for sensor in ['acc', 'gyro'] for axis in ['x', 'y', 'z']
                    ] + [
                        f"{sensor}_{axis}_median" for sensor in ['acc', 'gyro'] for axis in ['x', 'y', 'z']
                    ] + [
                        f"{sensor}_{axis}_min" for sensor in ['acc', 'gyro'] for axis in ['x', 'y', 'z']
                    ] + [
                        f"{sensor}_{axis}_max" for sensor in ['acc', 'gyro'] for axis in ['x', 'y', 'z']
                    ] + [
                        f"{sensor}_{axis}_signal_var" for sensor in ['acc', 'gyro'] for axis in ['x', 'y', 'z']
                    ] + [
                        f"{sensor}_{axis}_zero_crossings" for sensor in ['acc', 'gyro'] for axis in ['x', 'y', 'z']
                    ] + [
                        f"{sensor}_{axis}_zero_crossing_variance" for sensor in ['acc', 'gyro'] for axis in
                        ['x', 'y', 'z']
                    ] + [
                        f"{sensor}_{axis}_thd" for sensor in ['acc', 'gyro'] for axis in ['x', 'y', 'z']
                    ] + [
                        f"{sensor}_{axis}_spectral_entropy" for sensor in ['acc', 'gyro'] for axis in ['x', 'y', 'z']
                    ] + [
                        f"{sensor}_{axis}_dominant_freq" for sensor in ['acc', 'gyro'] for axis in ['x', 'y', 'z']
                    ] + [
                        f"{sensor}_{axis}_envelope_mean" for sensor in ['acc', 'gyro'] for axis in ['x', 'y', 'z']
                    ] + [
                        f"{sensor}_{axis}_slope_mean" for sensor in ['acc', 'gyro'] for axis in ['x', 'y', 'z']
                    ] + [
                        f"{sensor}_{axis}_trajectory" for sensor in ['acc', 'gyro'] for axis in ['x', 'y', 'z']
                    ] + [
                        f"{sensor}_{axis}_wl" for sensor in ['acc', 'gyro'] for axis in ['x', 'y', 'z']
                    ] + [
                        f"{sensor}_{axis}_cycle_count_peaks" for sensor in ['acc', 'gyro'] for axis in ['x', 'y', 'z']
                    ] + [
                        f"{sensor}_{axis}_pos_to_neg" for sensor in ['acc', 'gyro'] for axis in ['x', 'y', 'z']
                    ] + [
                        f"{sensor}_{axis}_num_transitions" for sensor in ['acc', 'gyro'] for axis in ['x', 'y', 'z']
                    ] + [
                        f"{sensor}_{axis}_second_harmonic_ratio" for sensor in ['acc', 'gyro'] for axis in
                        ['x', 'y', 'z']
                    ] + [
                        f"{sensor}_{axis}_low_freq_power" for sensor in ['acc', 'gyro'] for axis in ['x', 'y', 'z']
                    ] + [
                        f"acc_{axis}_lle" for axis in ['x', 'y', 'z']
                    ] + [
                        f"{sensor}_xy_correlation" for sensor in ['acc', 'gyro']
                    ] + [
                        f"{sensor}_xz_correlation" for sensor in ['acc', 'gyro']
                    ] + [
                        f"{sensor}_yz_correlation" for sensor in ['acc', 'gyro']
                    ] + [
                        f"{sensor}_circularity_xy" for sensor in ['acc', 'gyro']
                    ] + [
                        f"{sensor}_circularity_xz" for sensor in ['acc', 'gyro']
                    ] + [
                        f"{sensor}_circularity_yz" for sensor in ['acc', 'gyro']
                    ] + [
                        "mag_magnitude_mean",
                        "mag_magnitude_std",
                    ] + [
                        f"mag_{axis}_signal_var" for axis in ['x', 'y', 'z']
                    ]
    # Save feature_names
    save_json(feature_names, os.path.join(intermediate_output_dir, 'feature_names.json'))

    # Step 4: Split data into training and test sets (using Split2 logic)
    data_split = split_data(X_features_all, Y_vector_all, X_matrix_all)

    # --- Prepare full training data (100%) for learning all preprocessing parameters ---
    print("Learning GLOBAL preprocessing parameters from 100% of training data (Split 2 logic)...")
    X_train_full_preprocessed, norm_params_global = _apply_preprocessing_steps(
        data_split['X_train'], data_split['Y_train'], data_split['train_group_ids'], 'training',
        apply_fit=True, feature_names=feature_names
    )
    # Save norm_params_global
    save_json(norm_params_global, os.path.join(intermediate_output_dir, 'norm_params_global.json'))

    # --- Subsampling training data for VETTING ONLY (for feature selection) ---
    sampling_fraction_for_vetting = 0.4

    # Subsample for vetting from the ALREADY PREPROCESSED full training data
    if data_split['X_train'].shape[0] > 0:
        X_vetting, _, Y_train_subsampled_for_vetting, _, _, _ = \
            train_test_split(X_train_full_preprocessed, data_split['Y_train'], data_split['train_group_ids'],
                             train_size=sampling_fraction_for_vetting, stratify=data_split['Y_train'], random_state=42)
        print(
            f"Sampling {sampling_fraction_for_vetting * 100:.0f}% of PREPROCESSED training data for vetting (new size: {len(X_vetting)} samples)")

        # Step 5: Vet features (using the subsampled, preprocessed data)
        X_vetting, feature_names_from_vetting = vet_features(X_vetting, Y_train_subsampled_for_vetting,
                                                             data_split['train_group_ids'], 'split2',
                                                             feature_names)

        # Step 6: Feature selection with MRMR
        print("\n===== Processing Split2 =====")
        X_switched, selected_indices = select_features_mrmr(X_vetting, Y_train_subsampled_for_vetting, 'split2',
                                                            feature_names_from_vetting, max_features=10,
                                                            miq_threshold=0.2)
        # Save selected_indices
        save_json(selected_indices, os.path.join(intermediate_output_dir, 'selected_indices.json'))

        # Step 7: Train model
        X_train_model_input = X_train_full_preprocessed[:, selected_indices]

        print("\n===== Training Model for Split2 (Internal Evaluation) =====")
        model_a = train_model(X_train_model_input, data_split['Y_train'])
        print("===== Model training completed =====")

        # Step 8: Evaluate model (Internal Test Set)
        all_figures = []

        print("\n===== Evaluating Model for Split2 (Internal Test Set) =====")
        X_test_preprocessed, _ = _apply_preprocessing_steps(
            data_split['X_test'], data_split['Y_test'], data_split['test_group_ids'], 'test_internal',
            normalization_params=norm_params_global, apply_fit=False, feature_names=feature_names
        )
        X_test_selected = X_test_preprocessed[:, selected_indices]

        roc_auc_a, sensitivity_a, cm_a, overall_accuracy_a = evaluate_model(
            model_a, X_test_selected, data_split['Y_test'], 'split2', data_split['test_group_ids'],
            norm_params_global, selected_indices, all_figures, X_matrix_all, data_split['test_idx'], output_dir
        )
        print("===== Internal Model evaluation completed =====")

        print(f"Displaying all {len(all_figures)} ROC plots for split2...")
        plt.show()

        # Close all figures to free memory
        for fig in all_figures:
            plt.close(fig)

        # --- NEW STEP: Retrain final model on ALL available labeled data (train + internal test) ---
        print("\n===== Retraining FINAL Model on ALL available labeled data (Train + Internal Test) =====")
        Y_vector_all_cleaned = Y_vector_all[Y_vector_all != -1]
        X_features_all_cleaned = X_features_all[Y_vector_all != -1]

        X_all_preprocessed_for_final_model, _ = _apply_preprocessing_steps(
            X_features_all_cleaned, Y_vector_all_cleaned, X_matrix_all[Y_vector_all != -1, 3],
            'full_dataset_for_final_training', normalization_params=norm_params_global, apply_fit=False,
            feature_names=feature_names
        )

        valid_selected_indices_for_final = [idx for idx in selected_indices if
                                            idx < X_all_preprocessed_for_final_model.shape[1]]
        if len(valid_selected_indices_for_final) != len(selected_indices):
            print(
                f"Warning: Some selected feature indices ({len(selected_indices) - len(valid_selected_indices_for_final)}) are out of bounds for the full dataset. Using available features only.")
        X_all_final_for_retraining = X_all_preprocessed_for_final_model[:, valid_selected_indices_for_final]

        final_model_for_lecturer_data = train_model(X_all_final_for_retraining, Y_vector_all_cleaned)
        print("===== FINAL Model retraining completed on all labeled data =====")

        # Save the final trained model
        joblib.dump(final_model_for_lecturer_data,
                    os.path.join(intermediate_output_dir, 'final_model_for_lecturer_data.joblib'))

    else:
        print("No training data available for Split 2. Skipping model training and evaluation.")
        # Ensure outputs are still created/initialized even if no training data
        joblib.dump(None, os.path.join(intermediate_output_dir, 'final_model_for_lecturer_data.joblib'))
        save_json([], os.path.join(intermediate_output_dir, 'selected_indices.json'))
        save_json({}, os.path.join(intermediate_output_dir, 'norm_params_global.json'))
        # X_matrix_all and best_window_size are saved outside this if block.


if __name__ == "__main__":
    main()