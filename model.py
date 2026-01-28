import pandas as pd
from xgboost import XGBClassifier
import joblib

# Create a more robust training set
data = {
    'attempts': [1, 2, 1, 20, 25, 30, 2, 1, 40, 50],
    'failed_attempts': [0, 1, 1, 20, 24, 30, 0, 1, 39, 50],
    'unique_users': [1, 1, 1, 1, 1, 5, 1, 1, 1, 10],
    'label': [0, 0, 0, 1, 1, 1, 0, 0, 1, 1] # 1 = Attack
}

train_df = pd.DataFrame(data)
X = train_df[['attempts', 'failed_attempts', 'unique_users']]
y = train_df['label']

# Train with high importance on the attack class
model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
model.fit(X, y)

joblib.dump(model, 'brute_force_model.pkl')
print("Model re-trained with explicit thresholds!")