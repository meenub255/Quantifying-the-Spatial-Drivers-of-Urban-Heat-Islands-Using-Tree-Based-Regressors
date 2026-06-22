import pandas as pd
import numpy as np
from synthetic_data_generator import generate_synthetic_data
from spatial_models import run_modeling_pipeline
from xai_explainer import compute_shap_explanations, plot_shap_summary, plot_shap_dependence
from visualizer import plot_spatial_features, plot_model_comparison, plot_spatial_shap

def main():
    print("======================================================================")
    print("Urban Heat Island (UHI) Explainable AI (XAI) Analytics Pipeline")
    print("======================================================================\n")
    
    # 1. Generate Synthetic Spatial Dataset
    print("[Step 1] Generating synthetic urban microclimate spatial data...")
    df = generate_synthetic_data(grid_size=100, seed=42)
    df.to_csv('urban_microclimate_data.csv', index=False)
    print(f"Dataset generated and saved. Shape: {df.shape}\n")
    
    # 2. Visualize Input Features and LST Spatially
    print("[Step 2] Creating spatial visualization of landscape features...")
    plot_spatial_features(df, grid_size=100, save_path="spatial_features.png")
    print("Spatial features map saved as 'spatial_features.png'.\n")
    
    # 3. Model Training & Evaluation (Standard Split vs Spatial Block Split)
    print("[Step 3] Running machine learning modeling pipeline...")
    features = ['NDVI', 'Building_Density', 'SVF', 'Albedo', 'Distance_to_Water', 'Elevation']
    target = 'LST'
    
    pipeline_results = run_modeling_pipeline(df, features, target)
    comparison_df = pipeline_results['comparison']
    
    # Plot model comparison metrics
    plot_model_comparison(comparison_df, save_path="model_comparison.png")
    print("Model comparison plot saved as 'model_comparison.png'.\n")
    
    # 4. Explainable AI Analysis using SHAP
    print("[Step 4] Computing SHAP explainability values...")
    # We will explain the XGBoost Spatial model as it handles complex non-linear structures well
    xgb_model = pipeline_results['models']['xgb_spatial']
    X_train_s, X_test_s, y_train_s, y_test_s = pipeline_results['data_splits']['spatial']
    
    # Combine spatial split train and test features for global dataset explanations, keeping indices aligned
    X_all = pd.concat([X_train_s, X_test_s]).sort_index()
    df_coords = df[['X', 'Y']].sort_index()
    
    # Compute SHAP
    shap_results = compute_shap_explanations(xgb_model, X_all, df_coords)
    shap_values_obj = shap_results['shap_values_obj']
    shap_df = shap_results['shap_df']
    
    # Save SHAP CSV
    shap_df.to_csv('shap_explanations.csv', index=False)
    print("SHAP explanation values saved to 'shap_explanations.csv'.")
    
    # Generate global SHAP summary plot (Beeswarm)
    plot_shap_summary(shap_values_obj, X_all, save_path="shap_summary_beeswarm.png")
    
    # Generate Spatial SHAP maps
    print("Plotting spatial distribution of SHAP contributions...")
    plot_spatial_shap(shap_df, grid_size=100, save_path="spatial_shap_maps.png")
    print("Spatial SHAP maps saved as 'spatial_shap_maps.png'.\n")
    
    # Generate Dependency plots for key drivers (NDVI and Building Density)
    print("Generating non-linear SHAP dependence plots...")
    plot_shap_dependence(shap_values_obj, 'NDVI', X_all, save_path="shap_dependence_ndvi.png")
    plot_shap_dependence(shap_values_obj, 'Building_Density', X_all, save_path="shap_dependence_building_density.png")
    
    print("\n======================================================================")
    print("Pipeline Execution Complete!")
    print("All datasets, model comparisons, and XAI maps have been successfully generated.")
    print("======================================================================")

if __name__ == '__main__':
    main()
