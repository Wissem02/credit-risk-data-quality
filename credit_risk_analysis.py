"""
Credit Risk Scoring with an Automated Data Quality Framework
Dataset: German Credit (1,000 loans, 20 attributes, 30% defaults)

Steps
1. Automated data quality checks (completeness, validity, uniqueness, outliers)
   - the framework is itself tested by injecting known errors into a copy
2. Default-rate analysis by borrower segment
3. Credit scoring model (logistic regression) with cost-sensitive cut-off
4. Risk bands: default rate per score band, for portfolio monitoring
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).parent
IMG = ROOT / "images"; IMG.mkdir(exist_ok=True)
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#bbb", "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.labelcolor": MUTED})

df = pd.read_csv(ROOT / "data" / "german_credit.csv")
df["default"] = (df["default"] == 2).astype(int)   # source coding: 1 = repaid, 2 = default

# ------------------------------------------------- 1. Data quality framework
RULES = {
    "months_loan_duration": (1, 120),
    "amount": (1, 1_000_000),
    "age": (18, 100),
    "installment_rate": (1, 4),
    "residence_history": (1, 4),
    "existing_credits": (1, 10),
    "dependents": (0, 10),
}
ALLOWED = {
    "checking_balance": {"< 0 DM", "1 - 200 DM", "> 200 DM", "unknown"},
    "telephone": {"yes", "none"},
    "foreign_worker": {"yes", "no"},
}


def run_quality_checks(data: pd.DataFrame) -> pd.DataFrame:
    """Return one row per check with the number of failing records."""
    rows = []
    for col in data.columns:
        rows.append(("Completeness", f"{col} not null", int(data[col].isna().sum())))
    for col, (lo, hi) in RULES.items():
        bad = (~data[col].between(lo, hi)) & data[col].notna()
        rows.append(("Validity", f"{col} in [{lo}, {hi}]", int(bad.sum())))
    for col, allowed in ALLOWED.items():
        bad = ~data[col].isin(allowed) & data[col].notna()
        rows.append(("Validity", f"{col} in allowed list", int(bad.sum())))
    rows.append(("Uniqueness", "no duplicate records", int(data.duplicated().sum())))
    rows.append(("Consistency", "default flag is 0/1", int((~data["default"].isin([0, 1])).sum())))
    # Outliers are flagged for review, not rejected
    q1, q3 = data["amount"].quantile([0.25, 0.75])
    rows.append(("Outlier (review)", "amount > Q3 + 3 x IQR", int((data["amount"] > q3 + 3 * (q3 - q1)).sum())))
    out = pd.DataFrame(rows, columns=["dimension", "check", "failures"])
    out["status"] = np.where(out["failures"] == 0, "PASS",
                             np.where(out["dimension"].str.startswith("Outlier"), "REVIEW", "FAIL"))
    return out


dq = run_quality_checks(df)
(ROOT / "outputs").mkdir(exist_ok=True)
dq.to_csv(ROOT / "outputs" / "data_quality_report.csv", index=False)
print(dq[dq["status"] != "PASS"].to_string(index=False) or "All checks passed")

# Test the framework itself: inject 4 known errors into a copy and make sure each is caught
test = df.copy()
test.loc[0, "age"] = 150
test.loc[1, "amount"] = np.nan
test.loc[2, "checking_balance"] = "??"
test = pd.concat([test, test.iloc[[3]]])
caught = run_quality_checks(test)
expected = {"age in [18, 100]", "amount not null", "checking_balance in allowed list", "no duplicate records"}
assert expected <= set(caught.loc[caught["status"] == "FAIL", "check"]), "DQ framework missed an injected error"
print("DQ framework self-test: all 4 injected errors detected")

# ------------------------------------------------------ 2. Segment analysis
df["duration_band"] = pd.cut(df["months_loan_duration"], [0, 12, 24, 36, 72],
                             labels=["<= 12 m", "13-24 m", "25-36 m", "> 36 m"])
seg_checking = df.groupby("checking_balance")["default"].mean().reindex(
    ["< 0 DM", "1 - 200 DM", "> 200 DM", "unknown"])
seg_duration = df.groupby("duration_band", observed=True)["default"].mean()
seg_history = df.groupby("credit_history")["default"].mean().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, s, t in [(axes[0], seg_checking, "By checking account balance"),
                 (axes[1], seg_duration, "By loan duration")]:
    ax.bar(s.index.astype(str), s.values * 100, color=[ORANGE if v == s.max() else BLUE for v in s.values], width=0.6)
    for i, v in enumerate(s.values):
        ax.text(i, v * 100 + 1, f"{v:.0%}", ha="center", fontsize=9, color=INK)
    ax.axhline(df["default"].mean() * 100, color=MUTED, ls="--", lw=1)
    ax.set_title(t, loc="left", color=INK, fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8)
axes[0].set_ylabel("Default rate (%)")
fig.tight_layout(); fig.savefig(IMG / "default_rate_segments.png", dpi=150); plt.close(fig)

# ------------------------------------------------------- 3. Scoring model
y = df["default"]
X = df.drop(columns=["default", "duration_band"])
num = ["months_loan_duration", "amount", "age", "installment_rate", "residence_history",
       "existing_credits", "dependents"]
cat = [c for c in X.columns if c not in num]
model = Pipeline([
    ("pre", ColumnTransformer([("num", StandardScaler(), num),
                               ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])),
    ("clf", LogisticRegression(max_iter=2000, C=0.5)),
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
auc = roc_auc_score(y, proba)
gini = 2 * auc - 1

# Cost-sensitive cut-off: approving a bad loan costs 5x more than rejecting a good one
thresholds = np.linspace(0.05, 0.95, 91)
costs = []
for t in thresholds:
    tn, fp, fn, tp = confusion_matrix(y, (proba >= t).astype(int)).ravel()
    costs.append(5 * fn + 1 * fp)
best_t = float(thresholds[int(np.argmin(costs))])
cost_default = costs[int(np.argmin(np.abs(thresholds - 0.5)))]
cost_best = min(costs)
tn, fp, fn, tp = confusion_matrix(y, (proba >= best_t).astype(int)).ravel()
print(f"AUC {auc:.3f} | Gini {gini:.3f} | best cut-off {best_t:.2f} | "
      f"cost at 0.50 = {cost_default}, at best = {cost_best}")

fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.plot(thresholds, costs, color=BLUE, lw=2)
ax.axvline(best_t, color=ORANGE, lw=1)
ax.text(best_t + 0.02, max(costs) * 0.9, f"optimal cut-off {best_t:.2f}", color=INK, fontsize=9)
ax.set_xlabel("Probability-of-default cut-off"); ax.set_ylabel("Misclassification cost")
ax.set_title("Choosing the approval cut-off (bad loan = 5x cost)", loc="left", color=INK,
             fontsize=11, fontweight="bold")
fig.tight_layout(); fig.savefig(IMG / "cost_curve.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------ 4. Risk bands
df["pd"] = proba
df["risk_band"] = pd.qcut(df["pd"], 5, labels=["A (lowest)", "B", "C", "D", "E (highest)"])
bands = df.groupby("risk_band", observed=True).agg(loans=("default", "size"),
                                                   default_rate=("default", "mean"),
                                                   avg_amount=("amount", "mean"))
print(bands.round(3))

fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.bar(bands.index.astype(str), bands["default_rate"] * 100, color=BLUE, width=0.6)
for i, v in enumerate(bands["default_rate"]):
    ax.text(i, v * 100 + 1, f"{v:.0%}", ha="center", fontsize=9, color=INK)
ax.set_ylabel("Observed default rate (%)")
ax.set_title("Default rate by risk band (model score quintiles)", loc="left", color=INK,
             fontsize=11, fontweight="bold")
fig.tight_layout(); fig.savefig(IMG / "risk_bands.png", dpi=150); plt.close(fig)

summary = {
    "rows": len(df), "default_rate": round(y.mean(), 4),
    "dq_checks": len(dq), "dq_pass": int((dq["status"] == "PASS").sum()),
    "dq_review": dq[dq["status"] == "REVIEW"][["check", "failures"]].to_dict(orient="records"),
    "dq_fail": dq[dq["status"] == "FAIL"][["check", "failures"]].to_dict(orient="records"),
    "seg_checking": seg_checking.round(4).to_dict(),
    "seg_duration": {str(k): round(v, 4) for k, v in seg_duration.items()},
    "seg_history": seg_history.round(4).to_dict(),
    "auc": round(auc, 3), "gini": round(gini, 3),
    "best_cutoff": round(best_t, 2), "cost_at_050": int(cost_default), "cost_at_best": int(cost_best),
    "cost_reduction_pct": round(100 * (cost_default - cost_best) / cost_default, 1),
    "recall_at_best": round(tp / (tp + fn), 3),
    "bands": {str(k): {"loans": int(r.loans), "default_rate": round(r.default_rate, 4)}
              for k, r in bands.iterrows()},
}
(ROOT / "results.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
