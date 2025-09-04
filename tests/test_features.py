import pandas as pd
from src.features.signal_features import SignalFeatureExtractor


def test_extract_all_features():
    df = pd.DataFrame({
        'id': ['1']*5,
        'datetime': pd.date_range('2020-01-01', periods=5, freq='s'),
        'X': [0,1,0,1,0], 'Y':[0,0,1,1,0], 'Z':[0,0,0,1,1],
        'EDA':[0.1,0.2,0.15,0.3,0.25], 'HR':[70,71,69,75,73], 'TEMP':[36.5,36.6,36.5,36.7,36.6],
        'label':[0,0,1,0,0]
    })
    fe = SignalFeatureExtractor()
    out = fe.extract_all_features(df)
    assert 'movement_magnitude' in out.columns
    assert 'EDA_mean_60s' in out.columns
    assert 'HR_variability' in out.columns