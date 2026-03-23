import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DiabetesPreprocessor:
    def __init__(self):
        self.encoders = {}

    def preprocess(self, path):
        df = pd.read_csv(path)
        df = df.replace('?', np.nan)

        drop_cols = ['encounter_id', 'patient_nbr', 'weight',
                     'payer_code', 'medical_specialty']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        df = df.dropna(thresh=len(df.columns) * 0.8)

        for col in df.select_dtypes(include=[np.number]):
            df[col] = df[col].fillna(df[col].median())

        for col in df.select_dtypes(include=['object']):
            if col != 'readmitted':
                df[col] = df[col].fillna(df[col].mode()[0])

        df['target'] = (df['readmitted'] == '<30').astype(int)
        df = df.drop(columns=['readmitted'])

        for col in df.select_dtypes(include=['object']):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le

        X = df.drop(columns=['target']).values
        y = df['target'].values

        return X, y
