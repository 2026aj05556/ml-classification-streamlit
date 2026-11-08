"""Train 5 classifiers on UCI Dry Bean dataset, save models and metrics."""
import io, json, urllib.request, zipfile
import joblib, pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from metrics import evaluate

URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL, timeout=60).read()))
data, _ = arff.loadarff(io.StringIO(z.read("DryBeanDataset/Dry_Bean_Dataset.arff").decode()))
df = pd.DataFrame(data)
df["Class"] = df["Class"].str.decode("utf-8") if df["Class"].dtype == object else df["Class"]
df.to_csv("data/dry_bean_full.csv", index=False)

X, y = df.drop(columns="Class"), df["Class"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
}
rows = []
for name, m in models.items():
    m.fit(Xtr, ytr)
    joblib.dump(m, f"model/{name.lower().replace(' ', '_')}.joblib")
    rows.append({"ML Model Name": name, **evaluate(m, Xte, yte)})
res = pd.DataFrame(rows).round(4)
res.to_csv("model/metrics_table.csv", index=False)
print(res.to_string(index=False))

