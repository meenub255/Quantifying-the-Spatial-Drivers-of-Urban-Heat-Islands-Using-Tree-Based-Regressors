import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def generate_synthetic_data(grid_size=100, seed=42):
    """
    Generates a synthetic spatial dataset representing an urban area of size grid_size x grid_size.
    Returns a pandas DataFrame with spatial features and Land Surface Temperature (LST).
    """
    np.random.seed(seed)
    
    # 1. Create Grid Coordinates
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Normalize coordinates to [0, 1] for calculations
    X_norm = X / grid_size
    Y_norm = Y / grid_size
    
    # 2. Define Urban Zones (e.g., commercial core/downtown, park, river)
    # Downtown is located around the center-right (0.6, 0.6)
    dist_to_downtown = np.sqrt((X_norm - 0.6)**2 + (Y_norm - 0.6)**2)
    
    # Park is located around (0.3, 0.4)
    dist_to_park = np.sqrt((X_norm - 0.3)**2 + (Y_norm - 0.4)**2)
    
    # River runs diagonally from top-left to bottom-right: Y_norm = X_norm
    dist_to_river = np.abs(Y_norm - X_norm) / np.sqrt(2)
    
    # 3. Generate Land Cover & Morphological Features
    # Building Density: High near downtown, low near park and river
    building_density = np.clip(1.0 - 1.8 * dist_to_downtown + 0.1 * np.random.randn(*X.shape), 0.0, 1.0)
    # Suppress building density in park and river
    building_density[dist_to_park < 0.15] *= 0.1
    building_density[dist_to_river < 0.08] *= 0.05
    # Smooth to represent spatial continuity
    building_density = gaussian_filter(building_density, sigma=1.0)
    
    # NDVI (Vegetation): High in park, medium in suburbs (away from downtown), low in downtown
    ndvi = np.clip(0.7 - 0.8 * building_density + 0.3 * np.exp(-dist_to_park**2 / 0.05) - 0.4 * np.exp(-dist_to_river**2 / 0.01) + 0.1 * np.random.randn(*X.shape), -0.1, 0.85)
    # Correct river zone NDVI (water has low NDVI)
    ndvi[dist_to_river < 0.04] = -0.05
    ndvi = gaussian_filter(ndvi, sigma=0.8)
    
    # SVF (Sky View Factor): Low in downtown (urban canyons), high in parks and river
    svf = np.clip(1.0 - 0.8 * building_density + 0.15 * np.random.randn(*X.shape), 0.1, 1.0)
    svf = gaussian_filter(svf, sigma=1.0)
    
    # Albedo: Dark asphalt in downtown, light surfaces/vegetation elsewhere
    albedo = np.clip(0.15 - 0.08 * building_density + 0.1 * ndvi + 0.05 * np.random.randn(*X.shape), 0.08, 0.35)
    albedo = gaussian_filter(albedo, sigma=0.5)
    
    # Distance to Water
    dist_to_water = dist_to_river
    
    # Elevation: Sloping from top-right (high) to bottom-left (low)
    elevation = 100.0 + 150.0 * (X_norm + Y_norm) + 10.0 * np.random.randn(*X.shape)
    elevation = gaussian_filter(elevation, sigma=2.0)
    
    # 4. Generate Land Surface Temperature (LST) with non-linear relationships
    # Base temperature
    lst_base = 28.0  # Celsius
    
    # Effects of features
    # - NDVI cooling (non-linear, saturating at high NDVI)
    ndvi_effect = -8.0 * np.tanh(2.5 * ndvi) 
    
    # - Building Density heating
    building_effect = 10.0 * (building_density ** 1.5)
    
    # - Sky View Factor cooling/ventilation effect
    svf_effect = -4.0 * svf
    
    # - Albedo cooling (reflective surfaces absorb less heat)
    albedo_effect = -12.0 * albedo
    
    # - River cooling buffer (strong cooling close to river, decays exponentially)
    river_effect = -5.0 * np.exp(-dist_to_water * 15.0)
    
    # - Elevation cooling lapse rate (-0.6C per 100m)
    elevation_effect = -0.006 * elevation
    
    # - Interaction term: building density has stronger heating effect when SVF is low (urban canyon heat trapping)
    interaction_effect = 4.0 * building_density * (1.0 - svf)
    
    # Combine effects
    lst = (lst_base + 
           ndvi_effect + 
           building_effect + 
           svf_effect + 
           albedo_effect + 
           river_effect + 
           elevation_effect + 
           interaction_effect)
    
    # Add spatially correlated noise (representing unobserved microclimate drivers like wind patterns)
    noise = np.random.normal(0, 1.5, size=X.shape)
    spatial_noise = gaussian_filter(noise, sigma=2.0)
    lst += spatial_noise
    
    # 5. Flatten arrays to construct DataFrame
    df = pd.DataFrame({
        'X': X.ravel(),
        'Y': Y.ravel(),
        'NDVI': ndvi.ravel(),
        'Building_Density': building_density.ravel(),
        'SVF': svf.ravel(),
        'Albedo': albedo.ravel(),
        'Distance_to_Water': dist_to_water.ravel(),
        'Elevation': elevation.ravel(),
        'LST': lst.ravel()
    })
    
    return df

if __name__ == '__main__':
    # Test generation and save
    df = generate_synthetic_data()
    df.to_csv('urban_microclimate_data.csv', index=False)
    print(f"Generated synthetic data with shape: {df.shape}")
    print(df.head())
    
    # Plot spatial maps of LST and NDVI to verify visually
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # LST Map
    lst_grid = df['LST'].values.reshape(100, 100)
    im1 = axes[0].imshow(lst_grid, cmap='inferno', origin='lower')
    axes[0].set_title('Land Surface Temperature (LST)')
    fig.colorbar(im1, ax=axes[0], label='°C')
    
    # NDVI Map
    ndvi_grid = df['NDVI'].values.reshape(100, 100)
    im2 = axes[1].imshow(ndvi_grid, cmap='YlGn', origin='lower')
    axes[1].set_title('NDVI (Vegetation)')
    fig.colorbar(im2, ax=axes[1], label='Index')
    
    plt.tight_layout()
    plt.savefig('synthetic_data_preview.png', dpi=150)
    print("Saved preview map to 'synthetic_data_preview.png'")
