# 🔬 AI-Based Forecasting of Single-Cell Morphokinetics for Personalized Sarcoma Treatment

This final project developed a complete computational pipeline that integrates image analysis (CellProfiler), data engineering, and a machine learning model (LSTM) to forecast the behavior of sarcoma cells under different treatments.

**Project Goal:** To investigate whether dynamic morphokinetic patterns—how cell shape and motion evolve over time—can serve as early indicators of metastatic potential and treatment response at the single-cell level.

## Project File Structure

The project files are organized according to the four main stages of the Cellomics and AI research workflow.

### 1. CellProfiler Pipeline Files

These files contain the image processing instructions used for segmentation and feature extraction.

| File | Description |
| :--- | :--- |
| **`finel_piplineE.cppipe`** | The final, runnable **CellProfiler** pipeline file (a binary file). Contains all modules used for sarcoma cell image analysis (Segmentation, Tracking, Measurement). |
| **`finel_piplineE.txt`** | A text version of the CellProfiler pipeline (for easy review and version control). |
| **`PipeLine_Finel.cpproj`** | The complete CellProfiler project file. |
| **`README – Tracking Tables & Analysis Code.txt`** | Documentation related to tracking and early data table generation. |
| **`README_CellProfile.txt`** | Specific documentation on the CellProfiler modules and parameters used. |

### 2. Segmentation, Tracking & Summary Table Processing

These files represent the core Data Engineering steps, including calculating movement metrics and cleaning the raw data, primarily using Python (PyBatch).

| File | Description |
| :--- | :--- |
| **`pybatch_code`** | The central Python code/script used for data processing, calculation of movement metrics, and merging. |
| **`SUMMARYTABLE_CLEANINGDATA`** | Script responsible for data cleaning, specifically identifying and filling missing time points and handling NaN values in the time-series data. |
| **`xls1`** | The supplementary Excel metadata file containing experimental details, used to map wells (B3, B6) to their specific treatments (DOX, HGF). |
| **`photoname_B6`, `photoname_B3`** | Files containing image names and `ImageNumber` sequence used to correctly synchronize experimental time points with the data. |
| **`Track_B6_1`, `Cells_B6_1`, `Cells_B3_1`, `Track_B3_1`** | Raw CSV measurement files exported directly from **CellProfiler** (per-object and tracking measurements). |
| **`Directly_Merged_...`, `Corrected_Updated_...`** | Intermediate tables where movement metrics (Speed, Distance, Acceleration) are calculated and merged into the main cell data. |
| **`SummeryTableAfterNaNHandling_B3/B6`** | Cleaned summary tables for individual batches (B3 and B6), ready for final analysis. |
| **`Combined_SummaryTable`** | The final, integrated DataFrame, combining data from both wells (B3 and B6) for subsequent LSTM analysis. |

### 3. Features Distribution & Descriptive Statistics

These files relate to the exploratory data analysis (EDA) and visualization of cellular features across treatments.

| File | Description |
| :--- | :--- |
| **`combined&normalized_table+distarbiation_bar.py`** | Python code used to generate feature distribution plots (Histograms, KDE) and Heatmaps to compare cell behavior over time between the two treatments. |
| **`combined&normalized_table+distarbiation bar README.txt`** | Documentation specific to the statistical analysis and visualization code. |
| **`all_forecast_metrics_summary_table (8).csv`** | Final summary table containing all computed LSTM forecasting metrics (MAE, RMSE, MAPE, P-Value) per cell. |

### 4. LSTM Forecasting

| File | Description |
| :--- | :--- |
| **`LSTM.py`** | The main execution file for the **LSTM** (Long Short-Term Memory) forecasting model. |
| **`lstm code.py`** | The complete source code for the LSTM model, including data preprocessing, feature selection (using Random Forest), training, and prediction logic. |
| **`README LSTM.txt`** | Specific documentation and notes for the LSTM model and its parameters. |
