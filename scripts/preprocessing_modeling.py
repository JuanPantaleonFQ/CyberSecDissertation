import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns


# === UTILITY FUNCTIONS ===
def step(title):
    print("\n" + "=" * 60)
    print(f"[STEP] {title}")
    print("=" * 60)

def info(msg):
    print(f"    ➤ {msg}")

def done():
    print("✅ [DONE]\n")

# === PIPELINE START ===
step("1. Loading Dataset")
csv_folder = 'dataset/CSV Files'
files = ['UNSW-NB15_1.csv', 'UNSW-NB15_2.csv', 'UNSW-NB15_3.csv', 'UNSW-NB15_4.csv']
df = pd.concat([pd.read_csv(os.path.join(csv_folder, f), header=None, low_memory=False) for f in files], ignore_index=True)
features_df = pd.read_csv(os.path.join(csv_folder, 'NUSW-NB15_features.csv'), encoding='ISO-8859-1')
df.columns = features_df['Name'].tolist()
info(f"Shape: {df.shape}")
done()

step("2. Preprocessing")
info("Dropping IP and port columns")
df.drop(['srcip', 'sport', 'dstip', 'dsport'], axis=1, inplace=True)

info("Encoding categorical features")
categorical_cols = ['proto', 'state', 'service']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

info("Splitting features and labels")
X = df.drop(columns=['Label'])
y = df['Label']

info("Cleaning missing data")
X.replace('', pd.NA, inplace=True)
X = X.apply(pd.to_numeric, errors='coerce')
X.fillna(X.mean(), inplace=True)
y = y.loc[X.index]
done()

step("3. Scaling & Splitting")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
done()

step("4. Training Decision Tree Classifier")
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
done()

step("5. Evaluating Model")
y_pred = clf.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(tabulate(cm, headers=['Predicted Normal', 'Predicted Attack'], showindex=['Actual Normal', 'Actual Attack'], tablefmt='fancy_grid'))

# Classification report
print("\nClassification Report:")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().round(3)
print(tabulate(report_df, headers='keys', tablefmt='pretty'))

# Creating a heatmap for the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Attack'], 
            yticklabels=['Normal', 'Attack'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Decision Tree')
plt.tight_layout()
plt.savefig("confusion_matrix.png")  # Save to file
plt.show()

done()
