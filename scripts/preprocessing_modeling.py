# UNSW-NB15 IDS Model - Preprocessing + Decision Tree Classifier
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

print("[INFO] Loading data...")
csv_folder = 'dataset/CSV Files'
files = ['UNSW-NB15_1.csv', 'UNSW-NB15_2.csv', 'UNSW-NB15_3.csv', 'UNSW-NB15_4.csv']

# Load all rows without assuming first row is a header
df = pd.concat([pd.read_csv(os.path.join(csv_folder, f), header=None, low_memory=False) for f in files], ignore_index=True)

# Load correct column names from features file
features_df = pd.read_csv(os.path.join(csv_folder, 'NUSW-NB15_features.csv'), encoding='ISO-8859-1')

# Check shape
print("Raw data columns:", df.shape[1])
print("Feature metadata rows:", features_df.shape[0])

# Assign the correct column names
column_names = features_df['Name'].tolist()  # Already includes label and attack_cat
df.columns = column_names


print("Loaded dataset shape:", df.shape)
print("Column names:", df.columns.tolist())


# 2. Drop irrelevant columns
print("[INFO] Dropping IP and port columns...")
df.drop(['srcip', 'sport', 'dstip', 'dsport'], axis=1, inplace=True)
 

 # 3. Encode categorical features
print("[INFO] Encoding categorical features...")
categorical_cols = ['proto', 'state', 'service']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# 4. Feature and label split
print("[INFO] Splitting features and labels...")
X = df.drop(columns=['Label'])
y = df['Label']


# 5. Scale features
print("[INFO] Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train-test split
print("[INFO] Splitting into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 7. Train model
print("[INFO] Training Decision Tree...")
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# 8. Predict and evaluate
print("[INFO] Evaluating model...")
y_pred = clf.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
