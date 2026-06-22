# Quantifying the Spatial Drivers of Urban Heat Islands Using Tree-Based Regressors and Explainable AI (XAI)

## 📌 Introduction & Project Overview

Rapid urbanization and global climate change have intensified the **Urban Heat Island (UHI)** effect—a microclimatic phenomenon where metropolitan areas experience significantly elevated surface and air temperatures compared to their rural surroundings. UHIs increase energy consumption for cooling, elevate greenhouse gas emissions, compromise thermal comfort, and raise heat-related mortality rates. Understanding and mitigating these localized thermal anomalies is a primary objective for contemporary urban planners and climatologists.

### The Scientific Challenge
Historically, analyzing the spatial drivers of UHIs relied on traditional linear statistical methods (such as Ordinary Least Squares or Geographically Weighted Regression). However, the relationships between Land Surface Temperature (LST) and urban features—such as vegetation cover (NDVI), urban morphology (Building Density, Sky View Factor), and material properties (Albedo)—are highly non-linear, threshold-dependent, and involve complex interactive dynamics. 

While advanced Machine Learning (ML) models like tree-based ensembles (Random Forest, XGBoost) excel at capturing these multi-dimensional, non-linear relationships with high predictive accuracy, they operate as **"black boxes."** This lack of interpretability prevents urban planners from extracting actionable, localized directives, such as: *At what threshold does increasing tree canopy coverage cease to yield additional cooling benefits?* or *How many degrees Celsius does high building density add to a specific neighborhood?*

Additionally, spatial datasets exhibit **spatial autocorrelation** (First Law of Geography: nearby things are more related than distant things). Applying standard random cross-validation yields data leakage, producing overoptimistic performance metrics that fail to generalize to unseen neighborhoods.

### The Solution: An Interpretable Spatial Framework
This repository presents a robust spatial data science framework to address these challenges. By combining high-performance tree-based regressors with **Explainable AI (XAI)** based on **SHAP (SHapley Additive exPlanations)**, this project:
1. Simulates a multi-zone urban microclimate grid with realistic spatial structures.
2. Formulates and implements a **Spatial Block Cross-Validation (Block CV)** scheme to evaluate true model generalization on unseen geographical areas.
3. Quantifies and maps local microclimate drivers at the pixel level (**Spatial SHAP**), turning black-box predictions into interpretable, spatially explicit policy recommendations.

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
