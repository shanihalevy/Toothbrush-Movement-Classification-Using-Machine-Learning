# main_05_predict_blind_data.py
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
import shutil  # For removing directories

# --- Set a global NumPy random seed for reproducibility ---
np.random.seed(42)

# Global group number for output files
GROUP_NUMBER = "05"


# --- Utility functions for saving/loading (copied from part 1 for consistency) ---

def save_numpy_to_csv(data, filename, output_dir):
    pd.DataFrame(data).to_csv(os.path.join(output_dir, filename), index=False)

def load_numpy_from_csv(filename, output_dir):
    return pd.read_csv(os.path.join(output_dir, filename)).values


def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


# --- Original functions relevant for prediction (copied as is) ---
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

############################## Part F: Vetting & Normalization (only _apply_preprocessing_steps is needed here) #########################################

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


# Main function for Part 2
def main():
    output_dir = os.path.dirname(__file__)
    intermediate_output_dir = os.path.join(output_dir, 'data', 'intermediate_outputs')
    if not os.path.exists(intermediate_output_dir):
        print(f"Error: Intermediate output directory not found at {intermediate_output_dir}")
        print("Please run main_05_train_and_prepare.py first to generate the necessary files.")
        return

    # Load variables from Part 1
    print("\nLoading saved variables from Part 1...")
    try:
        with open(os.path.join(intermediate_output_dir, 'best_window_size.txt'), 'r') as f:
            best_window_size = int(f.read().strip())  # Added .strip() for robustness

        norm_params_global = load_json(os.path.join(intermediate_output_dir, 'norm_params_global.json'))
        selected_indices = load_json(os.path.join(intermediate_output_dir, 'selected_indices.json'))
        feature_names = load_json(os.path.join(intermediate_output_dir, 'feature_names.json'))

        final_model_for_lecturer_data = joblib.load(
            os.path.join(intermediate_output_dir, 'final_model_for_lecturer_data.joblib'))

    except FileNotFoundError as e:
        print(f"Error loading required files: {e}")
        print("Ensure 'main_05_train_and_prepare.py' was run successfully and generated all intermediate outputs.")
        return
    except Exception as e:
        print(f"An error occurred while loading files: {e}")
        return

    # --- New section: Handling lecturer's blind_data (now as a regular directory) ---
    print("\nStarting processing for the given Final Test data - blind_data directory...")  # CHANGED MESSAGE

    # Define the path to the blind_data directory (now assumed to be pre-extracted)
    blind_data_dir_path = os.path.join(output_dir, 'data',
                                       'Blind_Data')  # CHANGED: pointing to the directory, not a zip file

    # Check if the blind_data directory exists
    if not os.path.isdir(blind_data_dir_path):  # CHANGED: check if directory exists
        print(f"Error: blind_data directory not found at path: {blind_data_dir_path}")  # CHANGED MESSAGE
        print(
            "Please ensure 'blind_data' is a regular (unzipped) folder inside the 'data' folder in the same directory as the Python file.")  # CHANGED MESSAGE
        # No temp_dir to clean up here, as we didn't create one yet for blind_data
        return

    # No extraction needed here, as blind_data is already a directory
    print(f"Using blind_data from directory: {blind_data_dir_path}")  # CHANGED MESSAGE

    # Segmentation and feature extraction for lecturer's data
    print(f"\nLoading Final Test data with window size {best_window_size} and extracting all features...")
    # Pass the path to the blind_data directory as the temp_dir to load_all_groups
    X_matrix_finaltest, Y_vector_finaltest, X_features_finaltest = load_all_groups(
        [blind_data_dir_path], window_size=best_window_size, features_to_extract='all'
        # CHANGED: passing the directory path
    )

    if X_matrix_finaltest.shape[0] == 0:
        print(
            "No data found in the given Final Test data - blind_data directory. Cannot generate predictions.")  # CHANGED MESSAGE
        # No temp_dir to clean up here for blind_data, it's the source dir
        return

    # Normalization of lecturer's data using GLOBAL parameters learned from pool_dataset training
    print("\nNormalizing Final Test data using GLOBAL parameters from internal training set...")
    X_features_finaltest_normalized, _ = _apply_preprocessing_steps(
        X_features_finaltest, Y_vector_finaltest, X_matrix_finaltest[:, 3], 'finaltest',
        normalization_params=norm_params_global, apply_fit=False, feature_names=feature_names
    )

    # Select features for lecturer's data using indices from MRMR (selected_indices)
    if final_model_for_lecturer_data is None:
        print(
            "Error: Final SVM model failed to load. Cannot make predictions on lecturer's data.")
        # No temp_dir to clean up here
        return

    if not selected_indices:
        print("Selected features failed to load. Cannot process lecturer's data.")
        # No temp_dir to clean up here
        return

    print("\nSelecting features for lecturer's data based on trained model's feature set...")
    valid_selected_indices = [idx for idx in selected_indices if idx < X_features_finaltest_normalized.shape[1]]
    if len(valid_selected_indices) != len(selected_indices):
        print(
            f"Warning: Some selected feature indices ({len(selected_indices) - len(valid_selected_indices)}) are out of bounds for the lecturer's data. Using available features only.")
    selected_indices = valid_selected_indices

    X_finaltest_selected = X_features_finaltest_normalized[:, selected_indices]

    print("\nMaking predictions on lecturer's data using the FINAL model...")
    predictions_finaltest = final_model_for_lecturer_data.predict(X_finaltest_selected)

    # Export predictions to CSV
    predictions_output_filename = os.path.join(output_dir, f"{GROUP_NUMBER}_predictions.csv")
    export_predictions_csv(X_matrix_finaltest, predictions_finaltest, predictions_output_filename)

    # Step 10: No cleanup of temp_dir_finaltest here, as it's now the source directory 'blind_data' itself,
    # which is not created by this script and should not be removed.
    print("\nPrediction complete. No temporary directories created by this script for blind_data were left behind.")


if __name__ == "__main__":
    main()