import glob, io, os, urllib.request
import joblib, numpy as np, pandas as pd, streamlit as st
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_curve, auc
from metrics import evaluate

st.set_page_config(page_title="Dry Bean Classifier Comparison", page_icon="🫘", layout="wide")
st.title("🫘 Dry Bean Classification – Model Comparison")
st.caption("ML Assignment 2 · UCI Dry Bean Dataset · 7 bean classes · 16 shape features")

# TODO: set to your repo's raw URL, e.g. https://raw.githubusercontent.com/2026aj05556/ml-classification-streamlit/main/data/dry_bean_full.csv
GITHUB_CSV_URL = "https://raw.githubusercontent.com/<user>/<repo>/main/data/dry_bean_full.csv"
FULL_CSV = os.path.join(os.path.dirname(__file__), "data", "dry_bean_full.csv")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
models = {os.path.basename(p)[:-7].replace("_", " ").title(): p
          for p in sorted(glob.glob(os.path.join(MODEL_DIR, "*.joblib")))}

@st.cache_resource
def load(path):
    return joblib.load(path)

with st.sidebar:
    st.header("Controls")
    choice = st.selectbox("Select model (for the single-model detail tab)", list(models))
    source = st.radio("Data source", ["Full dataset (data/dry_bean_full.csv)", "GitHub (dry_bean_full.csv)", "Upload CSV"])
    up = None
    if source == "Upload CSV":
        up = st.file_uploader("Upload test data (CSV)", type=["csv", "zip"],
                              help="Must contain the 16 feature columns and the 'Class' target column.")

@st.cache_data(show_spinner="Loading test data from GitHub...")
def from_github(url):
    return pd.read_csv(io.BytesIO(urllib.request.urlopen(url, timeout=10).read()))

if source == "Upload CSV":
    if up is None:
        st.info("⬅️ Upload a test CSV (features + `Class` column) to evaluate the saved models.")
        st.stop()
    df = pd.read_csv(up, compression="zip" if up.name.endswith(".zip") else "infer")
elif source.startswith("Full"):
    df = pd.read_csv(FULL_CSV)
    st.success("Loaded data/dry_bean_full.csv (13,611 rows). Note: ~80% of these rows were used for training, so scores are optimistic vs. the held-out test set.")
else:
    try:
        df = from_github(GITHUB_CSV_URL)
        st.success("Loaded dry_bean_full.csv directly from GitHub.")
    except Exception:
        st.warning("Could not reach the GitHub URL (set `GITHUB_CSV_URL` in app.py); using the bundled data/dry_bean_full.csv.")
        df = pd.read_csv(FULL_CSV)
model = load(models[choice])
features = list(model.feature_names_in_)
missing = [c for c in features + ["Class"] if c not in df.columns]
if missing:
    st.error(f"Uploaded CSV is missing required columns: {missing}")
    st.stop()

X, y = df[features], df["Class"]
st.write(f"**Data:** {len(df)} rows · {len(features)} features")

@st.cache_data(show_spinner="Running all 5 models...")
def run_all(data):
    rows = [{"ML Model Name": n, **evaluate(load(p), data[features], data["Class"])} for n, p in models.items()]
    return pd.DataFrame(rows).set_index("ML Model Name")

res = run_all(df)
winner = res.sort_values(["MCC", "F1"], ascending=False).index[0]

tab_sum, tab_sel = st.tabs(["🏆 All models – results & winner", f"🔍 Single model detail: {choice}"])

with tab_sel:
    st.dataframe(df.head(), width="stretch")
    m = res.loc[choice]
    st.subheader(f"Evaluation metrics – {choice}")
    if choice == winner:
        st.success("🏆 This model is the overall winner on this data.")
    for c, (k, v) in zip(st.columns(6), m.items()):
        c.metric(k, f"{v:.4f}")
    model = load(models[choice])
    pred = model.predict(X)
    labels = list(model.classes_)
    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y, pred, labels=labels), annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.subheader("Classification report")
        st.dataframe(pd.DataFrame(classification_report(y, pred, output_dict=True, zero_division=0)).T.round(3),
                     width="stretch")

    proba = model.predict_proba(X)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Normalized confusion matrix (recall per class)")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y, pred, labels=labels, normalize="true"), annot=True, fmt=".2f",
                    cmap="Greens", xticklabels=labels, yticklabels=labels, ax=ax, annot_kws={"size": 8})
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
    with c2:
        st.subheader("Per-class precision / recall / F1")
        rep = pd.DataFrame(classification_report(y, pred, output_dict=True, zero_division=0)).T.loc[labels,
                          ["precision", "recall", "f1-score"]]
        fig, ax = plt.subplots(figsize=(6, 5))
        rep.plot.bar(ax=ax, width=0.85, edgecolor="black")
        ax.set_ylim(0.4, 1.02); plt.xticks(rotation=45); ax.legend(fontsize=8)
        st.pyplot(fig)
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("ROC curve per class (one-vs-rest)")
        fig, ax = plt.subplots(figsize=(6, 5))
        for i, c in enumerate(labels):
            fpr, tpr, _ = roc_curve(y == c, proba[:, i])
            ax.plot(fpr, tpr, label=f"{c} ({auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.legend(fontsize=7, loc="lower right")
        st.pyplot(fig)
    with c4:
        st.subheader("Prediction confidence: correct vs wrong")
        conf = proba.max(axis=1); ok = pred == y.values
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(conf[ok], bins=20, alpha=0.7, label=f"Correct ({ok.sum()})", color="#16a34a")
        ax.hist(conf[~ok], bins=20, alpha=0.8, label=f"Wrong ({(~ok).sum()})", color="#dc2626")
        ax.set_xlabel("Max predicted probability"); ax.set_ylabel("Rows"); ax.legend()
        st.pyplot(fig)
    c5, c6 = st.columns(2)
    with c5:
        st.subheader("This model vs the others")
        fig, ax = plt.subplots(figsize=(6, 4))
        res.T.plot.bar(ax=ax, width=0.85, edgecolor="black", color=["#dc2626" if n == choice else "#9ca3af" for n in res.index]) if False else None
        sel = res.loc[choice]; best = res.max()
        pd.DataFrame({choice: sel, "Best of all models": best}).plot.bar(ax=ax, width=0.8, edgecolor="black",
                                                                       color=["#2563eb", "#facc15"])
        ax.set_ylim(0.6, 1.0); plt.xticks(rotation=0); ax.legend(fontsize=8)
        st.pyplot(fig)
    with c6:
        st.subheader("Correct vs wrong per class")
        cnt = pd.DataFrame({"Correct": pd.Series(y.values[ok]).value_counts(),
                            "Wrong": pd.Series(y.values[~ok]).value_counts()}).reindex(labels).fillna(0)
        fig, ax = plt.subplots(figsize=(6, 4))
        cnt.plot.bar(stacked=True, ax=ax, color=["#16a34a", "#dc2626"], edgecolor="black")
        plt.xticks(rotation=45); ax.set_ylabel("Rows")
        st.pyplot(fig)

with tab_sum:
    st.markdown(f"""<div style="background:#14532d;color:#fff;padding:22px;border-radius:12px;
border:3px solid #facc15;text-align:center;">
<div style="font-size:15px;letter-spacing:2px;">🏆 OVERALL WINNER</div>
<div style="font-size:36px;font-weight:800;">{winner}</div>
<div style="font-size:15px;">Accuracy {res.loc[winner,'Accuracy']:.4f} · F1 {res.loc[winner,'F1']:.4f} · MCC {res.loc[winner,'MCC']:.4f}</div>
</div>""", unsafe_allow_html=True)
    st.caption("Winner = highest MCC (ties broken by F1). Green cells = best model for that metric; the winner row is highlighted.")

    def style_row(row):
        return ["background-color:#facc15;color:#000;font-weight:700" if row.name == winner else "" for _ in row]
    st.subheader("Results for all models")
    st.dataframe(res.round(4).style.apply(style_row, axis=1)
                 .highlight_max(axis=0, color="#16a34a", props="color:white;font-weight:700")
                 .format("{:.4f}"), width="stretch")

    st.subheader("Metric comparison")
    fig, ax = plt.subplots(figsize=(10, 4))
    res.T.plot.bar(ax=ax, width=0.8, edgecolor="black")
    for bar_group, name in zip(ax.containers, res.index):
        if name == winner:
            for b in bar_group:
                b.set_edgecolor("#facc15"); b.set_linewidth(2.5)
    ax.set_ylim(0.6, 1.0); ax.set_ylabel("Score"); ax.legend(fontsize=8, ncol=3)
    plt.xticks(rotation=0)
    st.pyplot(fig)

    st.subheader("Ranking by MCC")
    rank = res[["MCC", "F1", "Accuracy"]].sort_values("MCC", ascending=False).reset_index()
    rank.insert(0, "Rank", [f"🥇 1" if i == 0 else str(i + 1) for i in range(len(rank))])
    st.dataframe(rank.round(4), hide_index=True, width="stretch")

    st.subheader("Metrics heatmap")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.heatmap(res, annot=True, fmt=".3f", cmap="YlGn", vmin=0.7, vmax=1.0, ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

    st.subheader("Radar chart – model profiles")
    ang = np.linspace(0, 2 * np.pi, len(res.columns), endpoint=False).tolist()
    fig = plt.figure(figsize=(6, 6)); ax = fig.add_subplot(111, polar=True)
    for n, r in res.iterrows():
        v = r.tolist()
        ax.plot(ang + ang[:1], v + v[:1], lw=3.5 if n == winner else 1.5, label=n + (" (winner)" if n == winner else ""))
    ax.set_xticks(ang); ax.set_xticklabels(res.columns); ax.set_ylim(0.6, 1.0)
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.1), fontsize=8)
    st.pyplot(fig)

    @st.cache_data(show_spinner="Computing per-class and ROC charts...")
    def per_model_details(data):
        out = {}
        for n, pth in models.items():
            mdl = load(pth); Xd, yd = data[features], data["Class"]
            out[n] = (mdl.predict(Xd), mdl.predict_proba(Xd), list(mdl.classes_))
        return out
    det = per_model_details(df)
    classes = det[winner][2]

    st.subheader("Per-class F1 score")
    f1c = pd.DataFrame({n: f1_score(y, d[0], average=None, labels=classes) for n, d in det.items()}, index=classes)
    fig, ax = plt.subplots(figsize=(10, 4)); f1c.plot.bar(ax=ax, width=0.85, edgecolor="black")
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("F1"); plt.xticks(rotation=0); ax.legend(fontsize=8, ncol=3)
    st.pyplot(fig)

    st.subheader("ROC curves (macro one-vs-rest) – all models")
    fig, ax = plt.subplots(figsize=(6, 5))
    grid = np.linspace(0, 1, 200)
    for n, (_, pr, cl) in det.items():
        tprs = []
        for i, c in enumerate(cl):
            fpr, tpr, _ = roc_curve(y == c, pr[:, i]); tprs.append(np.interp(grid, fpr, tpr))
        ax.plot(grid, np.mean(tprs, axis=0), lw=3 if n == winner else 1.5, label=f"{n} (AUC {res.loc[n, 'AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(fontsize=8, loc="lower right"); st.pyplot(fig)

    st.subheader("Confusion matrices – all models")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8)); axes = axes.ravel()
    for ax, (n, (pr, _, cl)) in zip(axes, det.items()):
        sns.heatmap(confusion_matrix(y, pr, labels=cl), annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=cl, yticklabels=cl, ax=ax, annot_kws={"size": 7})
        ax.set_title(n + (" (winner)" if n == winner else "")); ax.tick_params(labelsize=7)
    axes[-1].axis("off"); plt.tight_layout(); st.pyplot(fig)

    st.subheader("Class distribution of the data")
    fig, ax = plt.subplots(figsize=(8, 3)); y.value_counts().reindex(classes).plot.bar(ax=ax, color="#16a34a", edgecolor="black")
    plt.xticks(rotation=0); ax.set_ylabel("Rows"); st.pyplot(fig)

