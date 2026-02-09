🦷 Brush‑Sense: Sensor‑Based Toothbrush Movement Classifier
📘 Overview
Brush‑Sense is a Machine Learning project designed to classify four toothbrushing movements — Rest, Left‑Right, Up‑Down, Circular — using raw time‑series sensor data.
The project focuses on robust signal processing, feature engineering, and feature selection to transform noisy sensor data into a reliable classification model.

🧩 Project Structure
├── main05_train_and_prepare.py        # Training script, feature selection, artifact saving
├── main05_predict_blind_data.py       # Prediction script for unseen data
├── 05_README.txt                      # Technical reference documentation
└── data/
    ├── pool_dataset.zip               # Raw training sensor data
    ├── blind_data.zip                 # Raw test sensor data accessible only to the course staff; used for exam evaluation.
    └── intermediate_outputs/          # Saved model, feature metadata, normalization params


⚙️ Methodology & Pipeline
1. Data Preprocessing & Windowing

Cleaning raw time‑series sensor data
Segmenting into fixed‑size analysis windows
Saving global normalization parameters for consistent inference

2. Advanced Signal Processing & Feature Engineering

Extracting features from:

Time Domain: statistics, signal energy
Frequency Domain: PSD via Welch’s method, spectral entropy


Using Hilbert transforms and custom signal‑based descriptors

3. Rigorous Feature Selection

Two‑stage feature selection:

ReliefF → initial feature relevance ranking
MRMR → selecting non‑redundant, highly predictive features



4. Model Development

Training Support Vector Classifier (SVC) and Random Forest models
Final tuned model achieves high accuracy across all four movement classes


📊 Results Summary





















MetricValueModelSVC (Tuned)Accuracy~92.5% (example placeholder)Classes4 (Rest, L‑R, U‑D, Circular)
Actual metrics can be reproduced by loading the final model (final_model_for_lecturer_data.joblib) and running main05_predict_blind_data.py.
