import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for high-quality figures
sns.set_theme(style="ticks")
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 14
})

def plot_spatial_features(df, grid_size=100, save_path="spatial_features.png"):
    """
    Plots the spatial distribution of all predictor features and the target variable (LST).
    """
    features_to_plot = ['LST', 'NDVI', 'Building_Density', 'SVF', 'Albedo', 'Distance_to_Water']
    cmaps = {
        'LST': 'inferno',
        'NDVI': 'YlGn',
        'Building_Density': 'magma',
        'SVF': 'bone',
        'Albedo': 'copper',
        'Distance_to_Water': 'Blues_r'
    }
    titles = {
        'LST': 'Land Surface Temp (LST, °C)',
        'NDVI': 'Vegetation Index (NDVI)',
        'Building_Density': 'Building Density',
        'SVF': 'Sky View Factor (SVF)',
        'Albedo': 'Surface Albedo',
        'Distance_to_Water': 'Distance to Water'
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    axes = axes.ravel()
    
    for i, feature in enumerate(features_to_plot):
        grid = df[feature].values.reshape(grid_size, grid_size)
        im = axes[i].imshow(grid, cmap=cmaps[feature], origin='lower')
        axes[i].set_title(titles[feature], fontweight='semibold')
        axes[i].axis('off')
        
        # Colorbar
        cbar = fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        
    plt.suptitle("Urban Landscape Features and Temperature Distribution", y=0.98, fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved spatial features plot to {save_path}")

def plot_model_comparison(comparison_df, save_path="model_comparison.png"):
    """
    Plots a comparison of $R^2$ and RMSE scores between standard random split and spatial block split.
    """
    # Restructure data for plotting
    df_plot = comparison_df.reset_index().rename(columns={'index': 'Model_Config'})
    
    # Extract model and split type from config name
    df_plot['Model'] = df_plot['Model_Config'].apply(lambda x: x.split('_')[0])
    df_plot['Split_Type'] = df_plot['Model_Config'].apply(lambda x: x.split('_')[1])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # R2 Comparison
    sns.barplot(x='Model', y='Test_R2', hue='Split_Type', data=df_plot, ax=axes[0], palette='muted')
    axes[0].set_title("Test R² Score Comparison (Higher is Better)", fontweight='semibold')
    axes[0].set_ylabel("R² Score")
    axes[0].set_ylim(0, 1.05)
    
    # RMSE Comparison
    sns.barplot(x='Model', y='Test_RMSE', hue='Split_Type', data=df_plot, ax=axes[1], palette='muted')
    axes[1].set_title("Test RMSE Comparison (Lower is Better)", fontweight='semibold')
    axes[1].set_ylabel("RMSE (°C)")
    
    for ax in axes:
        ax.set_xlabel("Model Architecture")
        ax.legend(title="Validation Scheme")
        sns.despine(ax=ax)
        
    plt.suptitle("Influence of Spatial Autocorrelation on Model Validation", y=0.98, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved model comparison plot to {save_path}")

def plot_spatial_shap(shap_df, grid_size=100, save_path="spatial_shap_maps.png"):
    """
    Plots spatial maps of SHAP values to visualize where variables have positive or negative impacts on UHI.
    """
    shap_cols = [col for col in shap_df.columns if col.startswith('SHAP_')]
    
    # Exclude coordinate SHAP if present
    shap_cols = [col for col in shap_cols if col not in ['SHAP_X', 'SHAP_Y', 'SHAP_Elevation']]
    
    fig, axes = plt.subplots(2, int(np.ceil(len(shap_cols)/2)), figsize=(15, 9.5))
    axes = axes.ravel()
    
    # Find symmetric vmin/vmax for divergent colormap to properly center at 0 (neutral impact)
    max_val = max(shap_df[shap_cols].abs().max())
    
    for i, col in enumerate(shap_cols):
        grid = shap_df[col].values.reshape(grid_size, grid_size)
        
        # We use a diverging colormap: blue = cooling (negative SHAP), red = warming (positive SHAP)
        im = axes[i].imshow(grid, cmap='RdYlBu_r', origin='lower', vmin=-max_val, vmax=max_val)
        
        feature_name = col.replace('SHAP_', '')
        axes[i].set_title(f"Contribution of {feature_name} to LST", fontweight='semibold')
        axes[i].axis('off')
        
        # Colorbar
        cbar = fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        cbar.set_label('Temp Contribution (°C)', fontsize=8)
        cbar.ax.tick_params(labelsize=8)
        
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle("Spatial SHAP Maps: Quantifying Local Microclimate Drivers", y=0.98, fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved spatial SHAP maps to {save_path}")
