import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from preprocessing import preprocess_data

# Load data
df = pd.read_csv('data/churn.csv')

# Preprocess
X, y, scaler = preprocess_data(df)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('models/model.pkl', 'wb'))
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))

print("Model trained and saved!")