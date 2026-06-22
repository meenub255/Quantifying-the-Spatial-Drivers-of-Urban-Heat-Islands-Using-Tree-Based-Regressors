import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def spatial_block_split(df, grid_size=100, block_size=20, test_size=0.2, seed=42):
    """
    Splits the spatial grid into blocks to perform Spatial Cross-Validation.
    This prevents spatial autocorrelation from causing overoptimistic performance.
    """
    np.random.seed(seed)
    
    # Calculate block coordinates
    df['block_col'] = df['X'] // block_size
    df['block_row'] = df['Y'] // block_size
    df['block_id'] = df['block_col'] + (grid_size // block_size) * df['block_row']
    
    unique_blocks = df['block_id'].unique()
    num_test_blocks = int(len(unique_blocks) * test_size)
    
    test_blocks = np.random.choice(unique_blocks, size=num_test_blocks, replace=False)
    
    train_df = df[~df['block_id'].isin(test_blocks)].copy()
    test_df = df[df['block_id'].isin(test_blocks)].copy()
    
    # Drop temp block helper columns
    for d in [train_df, test_df]:
        d.drop(columns=['block_col', 'block_row', 'block_id'], inplace=True)
        
    return train_df, test_df

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Evaluates a model and returns performance metrics.
    """
    # Fit model
    model.fit(X_train, y_train)
    
    # Predict
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    # Metrics
    metrics = {
        'Train_R2': r2_score(y_train, pred_train),
        'Train_RMSE': np.sqrt(mean_squared_error(y_train, pred_train)),
        'Train_MAE': mean_absolute_error(y_train, pred_train),
        'Test_R2': r2_score(y_test, pred_test),
        'Test_RMSE': np.sqrt(mean_squared_error(y_test, pred_test)),
        'Test_MAE': mean_absolute_error(y_test, pred_test),
    }
    
    return model, metrics

def run_modeling_pipeline(df, features, target):
    """
    Runs the model training pipeline for Random Forest and XGBoost,
    comparing Standard Split and Spatial Block Split.
    """
    X = df[features]
    y = df[target]
    
    print("--- 1. Evaluating Standard Random Split (Non-Spatial) ---")
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    rf_rand = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    xgb_rand = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    _, rf_rand_metrics = evaluate_model(rf_rand, X_train_r, y_train_r, X_test_r, y_test_r, "RF (Random Split)")
    _, xgb_rand_metrics = evaluate_model(xgb_rand, X_train_r, y_train_r, X_test_r, y_test_r, "XGB (Random Split)")
    
    print("--- 2. Evaluating Spatial Block Split ---")
    train_df_s, test_df_s = spatial_block_split(df, block_size=20, test_size=0.2, seed=42)
    
    X_train_s = train_df_s[features]
    y_train_s = train_df_s[target]
    X_test_s = test_df_s[features]
    y_test_s = test_df_s[target]
    
    rf_spatial = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    xgb_spatial = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    rf_spatial, rf_s_metrics = evaluate_model(rf_spatial, X_train_s, y_train_s, X_test_s, y_test_s, "RF (Spatial Split)")
    xgb_spatial, xgb_s_metrics = evaluate_model(xgb_spatial, X_train_s, y_train_s, X_test_s, y_test_s, "XGB (Spatial Split)")
    
    # Consolidate results
    results = {
        'RF_Random': rf_rand_metrics,
        'XGB_Random': xgb_rand_metrics,
        'RF_Spatial': rf_s_metrics,
        'XGB_Spatial': xgb_s_metrics
    }
    
    results_df = pd.DataFrame(results).T
    print("\n--- Model Performance Comparison ---")
    print(results_df.round(4))
    
    # Return best model trained on spatial split (XGBoost is typically preferred for SHAP if performance is equal or better)
    best_model = xgb_spatial if xgb_s_metrics['Test_R2'] >= xgb_rand_metrics['Test_R2'] else rf_spatial
    
    # Return everything needed for downstream explainability
    return {
        'models': {
            'rf_spatial': rf_spatial,
            'xgb_spatial': xgb_spatial,
            'rf_random': rf_rand,
            'xgb_random': xgb_rand
        },
        'data_splits': {
            'spatial': (X_train_s, X_test_s, y_train_s, y_test_s),
            'random': (X_train_r, X_test_r, y_train_r, y_test_r)
        },
        'comparison': results_df
    }

if __name__ == '__main__':
    # Simple test run
    from synthetic_data_generator import generate_synthetic_data
    df = generate_synthetic_data()
    features = ['NDVI', 'Building_Density', 'SVF', 'Albedo', 'Distance_to_Water', 'Elevation']
    target = 'LST'
    run_modeling_pipeline(df, features, target)
