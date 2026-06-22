import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

def compute_shap_explanations(model, X, df_coordinates=None):
    """
    Computes SHAP values for the given tree-based model and features.
    If coordinates are provided, compiles spatial SHAP values.
    """
    print("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    print("Computing SHAP values...")
    shap_values_obj = explainer(X)
    
    # Extract values array (handles shap Explanation objects)
    if isinstance(shap_values_obj, shap.Explanation):
        shap_values = shap_values_obj.values
    else:
        shap_values = shap_values_obj
        
    print(f"SHAP values computed. Shape: {shap_values.shape}")
    
    # Create a DataFrame of SHAP values
    shap_df = pd.DataFrame(shap_values, columns=[f"SHAP_{col}" for col in X.columns])
    
    # Merge coordinates back if available to enable spatial mapping of explanations
    if df_coordinates is not None:
        shap_df = pd.concat([df_coordinates.reset_index(drop=True), shap_df], axis=1)
        
    return {
        'explainer': explainer,
        'shap_values_obj': shap_values_obj,
        'shap_values': shap_values,
        'shap_df': shap_df
    }

def plot_shap_summary(shap_values_obj, X, save_path="shap_summary.png"):
    """
    Generates a SHAP beeswarm summary plot.
    """
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values_obj, show=False)
    plt.title("SHAP Feature Importance & Direction of Influence", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved SHAP summary beeswarm plot to {save_path}")

def plot_shap_dependence(shap_values_obj, feature_name, X, save_path=None):
    """
    Generates a SHAP dependence plot showing non-linear relationships.
    """
    plt.figure(figsize=(8, 5))
    shap.plots.scatter(shap_values_obj[:, feature_name], color=shap_values_obj[:, feature_name], show=False)
    plt.title(f"SHAP Dependence Plot for {feature_name}", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved SHAP dependence plot for {feature_name} to {save_path}")
    else:
        plt.show()

if __name__ == '__main__':
    # Test stub
    pass
