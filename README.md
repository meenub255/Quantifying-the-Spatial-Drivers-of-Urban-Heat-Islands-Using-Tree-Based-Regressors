# Quantifying the Spatial Drivers of Urban Heat Islands Using Tree-Based Regressors and Explainable AI (XAI)

This repository contains a complete, end-to-end spatial data science framework to predict, validate, and interpret the spatial drivers of the **Urban Heat Island (UHI)** effect. The framework utilizes high-performance Tree-Based Regressors combined with **SHAP (SHapley Additive exPlanations)** to extract actionable microclimate analytics for urban planning.

---

## 🌟 Key Features
- **Urban Spatial Data Simulator**: Generates realistic synthetic urban environments ($100 \times 100$ grid cells, $10,000$ points) featuring downtown high-rise canyons, sprawling suburban zones, green parks, and a cooling river corridor.
- **Autocorrelation-Aware Validation**: Implements both **Random Train-Test Split** and **Spatial Block Split (Block CV)** to expose how spatial autocorrelation leads to overoptimistic model assessment.
- **Tree-Based Ensemble Models**: Includes Random Forest and XGBoost regressors optimized for spatial prediction.
- **Explainable AI (XAI)**:
  - **Global Interpretability**: Beeswarm summary plots indicating feature importance and direction of influence.
  - **Non-linear Relationship Identification**: SHAP dependence scatter plots displaying critical ecological thresholds (e.g., vegetation cooling limits).
  - **Local Interpretability (Spatial SHAP)**: Grid-projected maps illustrating exactly where and by how many degrees specific features (e.g., NDVI, building density, SVF) heat or cool the city.

---

## 📂 Project Structure
- `synthetic_data_generator.py`: Generates the spatial city grid and Land Surface Temperature (LST) target variable.
- `spatial_models.py`: Handles model training, evaluation, and compares random splits with Spatial Block validation.
- `xai_explainer.py`: Computes SHAP values and outputs summary/dependence plots.
- `visualizer.py`: Renders feature maps, model performance comparisons, and spatial SHAP grids.
- `main.py`: Automates the entire analytics pipeline end-to-end.
- `UHI_XAI_Analysis.ipynb`: Interactive Jupyter Notebook containing walk-through code, outputs, and narrative.

---

## 📈 Results and Performance

### Validation Scheme Comparison
Standard random splits suffer from spatial data leakage due to proximity, creating misleadingly high accuracy metrics. Using a **Spatial Block Split** (splitting by geographical neighborhoods) provides a realistic estimation of performance on unseen areas:

| Configuration | Train $R^2$ | Test $R^2$ | Test RMSE (°C) | Test MAE (°C) |
| :--- | :---: | :---: | :---: | :---: |
| **RF (Random Split)** | 0.9994 | 0.9962 | 0.2661 | 0.2054 |
| **XGB (Random Split)** | 0.9993 | 0.9970 | 0.2363 | 0.1820 |
| **RF (Spatial Block Split)** | 0.9995 | **0.9844** | 0.4470 | 0.3381 |
| **XGB (Spatial Block Split)** | 0.9994 | **0.9876** | **0.3988** | **0.3010** |

*Interpretation: XGBoost evaluated on unseen spatial blocks yields an outstanding $R^2$ of **0.9876** and a root-mean-squared error of **0.3988°C**, demonstrating highly robust generalization.*

---

## 🧠 Explainable AI (XAI) Output Interpretations

### 1. Global Feature Driving Factors (Beeswarm Plot)
- **NDVI (Vegetation Index)**: Strongest negative contributor to LST, indicating it is the most dominant cooling agent.
- **Building Density**: Dominant positive contributor to LST, serving as the primary driver of the urban heat anomaly.
- **Sky View Factor (SVF)** and **Albedo**: High SVF (open skies/lack of canyon trapping) and high albedo (reflective surfaces) actively reduce local temperatures.

### 2. Threshold Cooling/Heating Effects (SHAP Dependence Plots)
- **Vegetation (NDVI) Threshold**: The cooling effect begins at $NDVI > 0.1$, increases sharply, and plateaus around $NDVI \approx 0.5$. This indicates that greening initiatives yield the highest thermal comfort efficiency when targeting a vegetation index of $0.5$, beyond which additional greening has diminishing cooling returns.
- **Building Density Threshold**: Surface temperatures increase slowly up to a building density of $0.4$, above which LST rises rapidly. This highlights the risk of high-density zoning without structural mitigations.

### 3. Spatial SHAP Maps (Localized Impact Mapping)
By projecting SHAP values back to coordinates:
- The **Urban Park** zone shows a deep blue cooling footprint (up to $-6\text{°C}$ contribution from `NDVI`).
- The **Downtown Core** displays a deep red heating footprint (up to $+8\text{°C}$ contribution from `Building_Density`).
- The **River Corridor** shows an exponential buffer effect cooling adjacent urban cells.

---

## 🚀 Quick Start & Usage

### 1. Requirements
Install the required dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn shap xgboost
```

### 2. Run the Full Pipeline
To generate the dataset, train models, calculate SHAP explanations, and save all visual maps, run:
```bash
python main.py
```

### 3. Run Interactively
Launch the Jupyter notebook to walk through the analysis step-by-step:
```bash
jupyter notebook UHI_XAI_Analysis.ipynb
```
