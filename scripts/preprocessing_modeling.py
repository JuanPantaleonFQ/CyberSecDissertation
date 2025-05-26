import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import sys
from tqdm import tqdm
from sklearn.tree import plot_tree

'''
This script preprocesses the UNSW-NB15 dataset, trains a Decision Tree Classifier,
 evaluates its performance, and visualizes the results. 
'''
# === PROJECT ROOT ===
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === OUTPUT DIRECTORY ===
output_dir = os.path.join(project_root, "output/ML-DecisionTree")
os.makedirs(output_dir, exist_ok=True)

# === UTILITY FUNCTIONS ===
def load_csv_with_progress(file_path, chunksize=100000):
    chunks = []
    total_rows = sum(1 for _ in open(file_path))  # Count lines (approximate total rows)
    
    with tqdm(total=total_rows, desc=f"📥 Loading {os.path.basename(file_path)}", unit="rows") as pbar:
        for chunk in pd.read_csv(file_path, header=None, chunksize=chunksize, low_memory=False):
            chunks.append(chunk)
            pbar.update(len(chunk))
    
    return pd.concat(chunks, ignore_index=True)
def step(title):
    print("\n" + "=" * 60)
    print(f"[STEP] {title}")
    print("=" * 60)

def info(msg):
    print(f"    \u27a4 {msg}")

def done():
    print("\u2705 [DONE]\n")

# === PIPELINE START ===
step("1. Loading Dataset")
csv_folder = os.path.join(project_root, 'dataset/CSV Files')
files = ['UNSW-NB15_1.csv', 'UNSW-NB15_2.csv', 'UNSW-NB15_3.csv', 'UNSW-NB15_4.csv']
df_list = []

df_list = []
for f in files:
    path = os.path.join(csv_folder, f)
    df_list.append(load_csv_with_progress(path))

df = pd.concat(df_list, ignore_index=True)
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

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=42)
info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
done()

step("4. Training Decision Tree Classifier")
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
done()

step("5. Evaluating Model")
y_pred = clf.predict(X_test)

step("6. Visualizing Decision Tree")

plt.figure(figsize=(20, 10))  # You can adjust size
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=['Normal', 'Attack'],
    filled=True,
    rounded=True,
    max_depth=3  # Visualize only the top 3 levels for clarity
)
plt.title("Decision Tree Classifier (Top Levels)")
plt.tight_layout()
tree_path = os.path.join(output_dir, "decision_tree_visualization.png")
plt.savefig(tree_path)
plt.close()

print(f"📁 Tree visualization saved to {tree_path}")
done()


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(tabulate(cm, headers=['Predicted Normal (0)', 'Predicted Attack (1)'], showindex=['Actual Normal (0)', 'Actual Attack (1)'], tablefmt='fancy_grid'))

# Classification report
print("\nClassification Report:")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().round(3)
print(tabulate(report_df, headers='keys', tablefmt='pretty'))

print("\n" + "=" * 60)
print("Interpretation")
print("=" * 60)
print("\nTrue Positives (TP): Correctly predicted attacks.")
print("True Negatives (TN): Correctly predicted normal traffic.")
print("False Positives (FP): Incorrectly predicted attacks (actually normal).")
print("False Negatives (FN): Incorrectly predicted normal traffic (actually attacks).")

# === Save visualizations ===
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Attack'], 
            yticklabels=['Normal', 'Attack'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Decision Tree')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

# === ROC Curve ===
y_scores = clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve - Decision Tree')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "roc_curve.png"))
plt.close()

# === Precision-Recall Curve ===
precision, recall, _ = precision_recall_curve(y_test, y_scores)
avg_prec = average_precision_score(y_test, y_scores)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label=f'AP = {avg_prec:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Decision Tree')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
plt.close()

# === Save classification report ===
report_path = os.path.join(output_dir, "classification_report.csv")
report_df.to_csv(report_path)
print(f"\U0001F4C1 Classification report saved to {report_path}")

# === Save summary metrics ===
metrics_summary = {
    "Accuracy": [round(clf.score(X_test, y_test), 4)],
    "ROC_AUC": [round(roc_auc, 4)],
    "Average_Precision": [round(avg_prec, 4)]
}
summary_df = pd.DataFrame(metrics_summary)
summary_path = os.path.join(output_dir, "metrics_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\U0001F4C1 Summary metrics saved to {summary_path}")

# === Save model ===
model_path = os.path.join(output_dir, "decision_tree_ids_model.joblib")
joblib.dump(clf, model_path)
print(f"\U0001F4C1 Model saved to {model_path}")

done()
