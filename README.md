# 🦷 Brush-Sense: Sensor-Based Toothbrush Movement Classifier

This project implements a complete Machine Learning pipeline for classifying four toothbrushing movements: Rest, Left-Right, Up-Down, and Circular. The solution integrates signal processing, feature engineering, feature selection, and supervised ML models to transform noisy sensor data into a reliable behavioral classifier.

## Project File Structure

### Main Files

| File | Description |
| --- | --- |
| `main05_train_and_prepare.py` | Script for training the final model, performing feature selection, and saving all artifacts. |
| `main05_predict_blind_data.py` | Script for loading the final trained model and predicting on unseen blind test data. |
| `05_README.txt` | Technical reference and legacy notes. |
| `pool_dataset.zip` | Raw training sensor data independently collected and curated. |
| `blind_data.zip` | Blind test data used for exam evaluation (restricted access). |
| `intermediate_outputs` | Saved model, feature metadata, normalization parameters. |


## ⚙️ Methodology and Pipeline

### 1. Data Preprocessing and Windowing
- Cleaned raw sensor data  
- Segmented time-series into fixed-size analysis windows  
- Computed and saved global normalization parameters  

### 2. Signal Processing and Feature Engineering
Features extracted from:
- Time domain: statistical metrics, signal energy  
- Frequency domain: PSD using Welch’s method, spectral entropy  
- Advanced transforms: Hilbert-based features and custom descriptors  

### 3. Feature Selection
- ReliefF for initial relevance ranking  
- MRMR for selecting a compact, predictive subset of features  

### 4. Model Development
- Trained and tuned SVC and Random Forest models to evaluate which algorithm performed best.  
- Achieved high accuracy across all four brushing movements  

## 📊 Results Summary

| Metric | Value |
| --- | --- |
| Chosen Model | SVC (Tuned) |
| Accuracy | ~92.5% |
| Classes | Rest, Left-Right, Up-Down, Circular |

Actual metrics can be reproduced by running `main05_predict_blind_data.py` with the saved model (`final_model_for_lecturer_data.joblib`).

## 💻 Running the Project

### Install requirements:
pip install -r requirements.txt

### Run predictions:
python main05_predict_blind_data.py

The output file `05_predictions.csv` will be saved in the `data/` directory.


