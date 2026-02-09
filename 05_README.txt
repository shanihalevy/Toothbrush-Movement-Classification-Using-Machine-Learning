Group 05 README : Execution Guide

This document explains the project structure, the division into two main code files, and how to run the prediction code on the blind test data.

---

## 1. Project Structure

The project is organized in the following directory structure:
05_test.ZIP/
├── main_05_train_and_prepare.py  # Part 1 Code: Training and Data Preparation (Already executed by us)
├── main_05_predict_blind_data.py # Part 2 Code: Prediction on Test Data (To be run by the lecturer)
└── data/
├── pool_dataset.zip     # ZIP file containing the original training data (Pool Data)
├── blind_data.zip       # ZIP file containing the test data (formerly 'blind_data.zip')
└── intermediate_outputs/
├── best_window_size.txt            # Optimal window size selected (saved as an int)
├── norm_params_global.json         # Global normalization parameters (JSON dictionary)
├── selected_indices.json           # Indices of selected features (JSON list)
├── feature_names.json              # Names of all original features (JSON list)
├── X_matrix_all.csv                # Window metadata matrix from Pool Data (CSV)
├── final_model_for_lecturer_data.joblib # The final trained model (Joblib)
└── temp_dir_pool_path.txt          # Path to the temporary directory used for Pool Data extraction

* **`your_project_root/`**: This is the base directory of the project.
* **`data/`**: This directory contains all required data files.
    * **`pool_dataset.zip`**: The data file used for model training and refinement.
    * **`blind_data.zip`**: The data file on which predictions should be made.
    * **`intermediate_outputs/`**: This directory was created and populated by `train_and_prepare.py`. It contains all files that are the OUTPUTS of Code 1 and the INPUTS of Code 2.

---

## 2. Code Division and Executed Steps

The project has been divided into two main code files to separate the training and preparation phase from the prediction phase on new data:

### `main_05_train_and_prepare.py` (Part 1 Code) - **Already executed by us.**

This file is responsible for the entire End-to-End process using the `pool_dataset.zip` data:

1.  **Segmentation:** Divides signals into time windows (optimal window size is automatically selected).
2.  **Feature Extraction:** Extracts a wide range of features from each window, **including comprehensive handling of NaN and Inf values.**
3.  **Correlation Analysis (THD Correlation):** Evaluates correlation using THD (Total Harmonic Distortion) to determine an optimal window size (enforced to be 3 seconds).
4.  **Data Splitting:** Splits the `pool_dataset` data into a training set and an internal test set (using Split 2 method - 80% of groups for training, 20% for testing). Undefined labels (-1) were removed from the training set.
5.  **Vetting & Preprocessing:**
    * Handles NaN/Inf values and performs Outlier Capping.
    * Performs global normalization of features (Standard Scaling) based on the full training set.
    * Removes highly correlated features (Spearman > 0.8) while retaining the feature with higher Mutual Information.
    * Selects the top 20 features using ReliefF.
6.  **MRMR (Minimum Redundancy Maximum Relevance) Feature Selection:** From the features filtered in the Vetting stage, the top 10 features were selected using the MRMR method.
7.  **Model Training:**
    * An SVM model was trained on the split training set (Split 2 Training Set). Grid Search was performed to select the best hyperparameters.
    * An internal evaluation of the model was performed on the Split 2 Internal Test Set, and ROC graphs were generated .
    * **Finally, the model was re-trained on all available labeled data from `pool_dataset` (training + internal test, after removing -1 labels). This is the final model that will be used for prediction on the test data.**

**Outputs of Code 1 (Inputs for Code 2):**

The `main_05_train_and_prepare.py` code saved the following variables in the `data/intermediate_outputs/` directory, which will serve as INPUTS for the `main_05_predict_blind_data.py` code:

* **`best_window_size.txt`**: The optimal window size (a single numerical value).
* **`norm_params_global.json`**: A dictionary containing the mean and standard deviation of the features used for normalization. The 'all' key contains lists of means and standard deviations for each feature.
* **`selected_indices.json`**: A list of integer indices representing the feature columns selected after the MRMR process.
* **`feature_names.json`**: A list of all 186 original feature names specified in the code.
* **`X_matrix_all.csv`**: The full window metadata matrix from `pool_dataset` (for your **reference only**; not loaded by Code 2).
* **`Y_vector_all.csv`**: The full window metadata matrix from `pool_dataset` (for your **reference only**; not loaded by Code 2).
* **`X_features_all.csv`**: The full feature matrix extracted from `pool_dataset` (for your **reference only**; not loaded by Code 2).
* **`final_model_for_lecturer_data.joblib`**: The final trained model (`sklearn.svm.SVC` object) saved in `joblib` format.
* **`temp_dir_pool_path.txt`**: The path to the temporary directory where `pool_dataset.zip` was extracted. This directory will be deleted by Code 2.
* `05_expected_accuracy.csv`: The overall accuracy of the final model evaluated on the internal test set (CSV).  Located in the project's base directory (05_test/).


### `main_05_predict_blind_data.py` (Part 2 Code) - **To be run by the lecturer.**

This file is designed to perform predictions on the test data (`blind_data.zip`) blindly, using the model and parameters learned in Part 1.

**Functions included in Code 2 (and their inputs):**

* **`segment_signal(data_path, window_size)`**: Takes the path to the raw data directory (`data_path` - for `blind_data.zip` after extraction) and `window_size` (loaded from `best_window_size.txt`).
* **`extract_features(data_path, X_matrix, features_to_extract)`**: Takes the path to the raw data directory (`data_path`), the `X_matrix` (window metadata matrix from `blind_data.zip`), and `features_to_extract='all'`. It internally uses the feature names loaded from `feature_names.json` for NaN/Inf handling.
* **`load_all_groups(temp_dirs, window_size, features_to_extract)`**: A general function for loading data from groups, which uses the `segment_signal` and `extract_features` functions. It receives a list of paths to temporary directories (the `blind_data.zip` directory after extraction), `window_size` (from `best_window_size.txt`), and `features_to_extract='all'`.
* **`_apply_preprocessing_steps(X_data, Y_data, group_ids, split_type, normalization_params, apply_fit=False, feature_names=None)`**: A preprocessing function. For Code 2, `apply_fit` will be `False`, and `normalization_params` will be the parameters loaded from `norm_params_global.json`. It uses `feature_names` loaded from `feature_names.json` for logging purposes.
* **`export_predictions_csv(X_matrix, predictions, output_filename)`**: Takes the `X_matrix` (window metadata matrix from `blind_data.zip`), the model's predictions (`predictions`), and the output filename.

**Explanations for the functions themselves (e.g., what `calculate_thd` or `select_features_mrmr` does) are fully provided within the code files as detailed docstrings. This README focuses on understanding the execution process and dependencies between the files.**

---

## 3. Lecturer's Execution Instructions

**Step 1: Ensure all files and directories are in place.**

* Verify that the code files (`main_05_train_and_prepare.py`, `main_05_predict_blind_data.py`) are in the project's base directory.
* Verify that the `data/` directory exists in the same base directory.
* Verify that the ZIP files (`pool_dataset.zip`, `blind_data.zip`) are inside the `data/` directory.
* Verify that the `data/intermediate_outputs/` directory exists and contains all files mentioned in the "Outputs of Code 1" section above. **This directory and its files were generated by us after running `main_05_train_and_prepare.py`.**

**Step 2: Run Code 2 (Predict Blind Data).**
python main_05_predict_blind_data.py

## blind_data Data Format

As expected and per the instructions, the structure of the `blind_data.zip` file should be identical to `pool_data.zip`. Specifically:
* The `blind_data.zip` file contain additional subdirectories, each named after a group number (e.g., '01', '02', '03').
* Within each group subdirectory, all the necessary CSV files (sensor data and label files) for that specific group's recordings are expected to reside.

