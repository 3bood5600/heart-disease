"""Reusable preprocessing + inference utilities for Heart Disease model.
Assumes feature engineering steps from notebook 01.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Dict, Any

ENGINEERED_COLS = [
    'chol_per_age',
    'heart_rate_reserve',
    'risk_score'
]

RAW_NUMERIC = ['age','trestbps','chol','thalach','oldpeak','ca']
RAW_CATEGORICAL = ['sex','cp','fbs','restecg','exang','slope','thal']
ALL_RAW = RAW_NUMERIC + RAW_CATEGORICAL

# Minimal safe defaults for missing numeric values
DEFAULT_NUMERIC_FILL = {
    'age': 55,
    'trestbps': 130,
    'chol': 240,
    'thalach': 150,
    'oldpeak': 1.0,
    'ca': 0
}

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Feature engineering consistent with notebook 01
    with np.errstate(divide='ignore', invalid='ignore'):
        out['chol_per_age'] = out['chol'] / out['age']
    out['heart_rate_reserve'] = out['thalach'] - 70  # assume resting = 70 if not provided
    # Simple composite risk score (example): age + chol/50 + (exang * 5) + (oldpeak*2)
    out['risk_score'] = (
        out.get('age',0) + out.get('chol',0)/50.0 + out.get('exang',0).astype(float)*5 + out.get('oldpeak',0)*2
    )
    return out

def prepare_input(record: Dict[str, Any]) -> pd.DataFrame:
    """Validate, coerce types, add missing columns, engineer features."""
    df = pd.DataFrame([record])
    # Ensure all raw columns present
    for col in ALL_RAW:
        if col not in df.columns:
            df[col] = np.nan
    # Coerce numerics
    for col in RAW_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Fill numeric missing with defaults
    for col, val in DEFAULT_NUMERIC_FILL.items():
        df[col] = df[col].fillna(val)
    # Categorical as string
    for col in RAW_CATEGORICAL:
        df[col] = df[col].astype(str).fillna('0')
    df = engineer_features(df)
    return df

__all__ = ['prepare_input','engineer_features','ENGINEERED_COLS','ALL_RAW']
