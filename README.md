pip install pandas numpy scikit-learn matplotlib seaborn
import pandas as pd

# Load dataset (change path if needed)
df = pd.read_csv('creditcard.csv')

# Display basic info
print(df.head())
print(df['Class'].value_counts())  # 0 = normal, 1 = fraud
from sklearn.preprocessing import StandardScaler

# Drop irrelevant columns (if any)
# In creditcard.csv, the 'Time' and 'Amount' columns are often scaled
df['Amount'] = StandardScaler().fit_transform(df[['Amount']])
df = df.drop(['Time'], axis=1)

# Separate features and labels (for evaluation)
X = df.drop('Class', axis=1)
y = df['Class']
from sklearn.ensemble import IsolationForest

# Define model (contamination = approx. % of fraud cases)
model = IsolationForest(n_estimators=100, contamination=0.0017, random_state=42)
model.fit(X)

# Predict anomalies
predictions = model.predict(X)
# Convert output: 1 -> normal (0), -1 -> fraud (1)
df['Anomaly'] = [1 if x == -1 else 0 for x in predictions]
from sklearn.metrics import classification_report, confusion_matrix

print("Confusion Matrix:\n", confusion_matrix(y, df['Anomaly']))
print("\nClassification Report:\n", classification_report(y, df['Anomaly']))
import matplotlib.pyplot as plt
import seaborn as sns

# Plot fraud vs normal
plt.figure(figsize=(6, 4))
sns.countplot(x='Anomaly', data=df)
plt.title("Detected Fraudulent vs Normal Transactions")
plt.xlabel("Anomaly (1 = Fraud)")
plt.ylabel("Count")
plt.show()

