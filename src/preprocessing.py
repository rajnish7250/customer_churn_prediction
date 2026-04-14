import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    df = df.copy()

    # Drop unnecessary columns
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Convert target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Encode categorical
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Split features/target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler