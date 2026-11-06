"""
train_model.py — Sprint 4, step 3
----------------------------------
Trains alert-vs-drowsy classifiers on windows.csv and evaluates them
honestly:

  * Cross-validation is split BY PERSON (GroupKFold), so the model is
    always tested on people it never saw during training.
  * Results are compared with a coin-flip baseline and with the simple
    hand-written rules from Sprints 1-2 (no machine learning).

Usage:
    python train_model.py --windows windows.csv --out-dir results

Outputs (in --out-dir):
    report.md                 the full written report (open this first)
    cv_results.csv            metrics table for every model / baseline
    confusion_matrix.png      best model, out-of-fold predictions
    roc_curves.png            all models + the single-feature rules
    feature_importance.png    random forest, trained on all data
    drowsiness_model.joblib   best model + feature names + window settings
                              (Sprint 5 loads this for live inference)
"""

import argparse
import datetime
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")            # draw to files; works without a display (Kaggle etc.)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from window_features import REL_FEATURE_NAMES

META_COLS = ["video_id", "subject", "fold", "group", "label", "window_start", "ear_open_level"]
CLASS_NAMES = ["alert", "drowsy"]

# These depend on a "neutral head pose" taken from the FIRST 3 SECONDS of each video.
# In the live system the user is told to look straight ahead then; in dataset videos
# those seconds are arbitrary, so the features are left out unless asked for.
BASELINE_DEPENDENT = ["pitch_delta_mean", "pitch_delta_max", "yaw_delta_absmean",
                      "droop_frac", "turned_frac"]
# Eye features that use the fixed EAR threshold / raw EAR (not person-relative)
RAW_EYE_FEATURES = ["ear_mean", "ear_std", "ear_min", "ear_p10", "closed_frac", "perclos_last",
                    "closure_per_min", "closure_mean_dur", "closure_max_dur", "long_closures"]


def make_models(seed):
    imputer = lambda: SimpleImputer(strategy="median", keep_empty_features=True)
    return {
        "random_forest": Pipeline([
            ("impute", imputer()),
            ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                          n_jobs=-1, random_state=seed)),
        ]),
        "logistic_regression": Pipeline([
            ("impute", imputer()),
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000)),
        ]),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=4, random_state=seed),
    }


def score(y, p, threshold=0.5):
    pred = (np.asarray(p) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
    }


def bootstrap_ci(y, p, groups, n_boot=500, seed=0, threshold=0.5):
    """95% intervals for accuracy and ROC-AUC, resampling PEOPLE (not windows),
    because windows from the same person are not independent."""
    rng = np.random.default_rng(seed)
    people = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in people}
    accs, aucs = [], []
    for _ in range(n_boot):
        pick = rng.choice(people, size=len(people), replace=True)
        ii = np.concatenate([idx[g] for g in pick])
        yy, pp = y[ii], p[ii]
        accs.append(accuracy_score(yy, pp >= threshold))
        if len(np.unique(yy)) > 1:
            aucs.append(roc_auc_score(yy, pp))
    return (np.percentile(accs, [2.5, 97.5]),
            np.percentile(aucs, [2.5, 97.5]) if aucs else (np.nan, np.nan))


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:.3f}"
    return str(v)


def md_table(df, index_name="model"):
    cols = list(df.columns)
    lines = [f"| {index_name} | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        lines.append(f"| {idx} | " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def write_report(path, ctx):
    """Write a plain-Markdown report from the numbers computed in main()."""
    t = ctx["table"]
    best = ctx["best"]
    b = t.loc[best]
    rules = [r for r in t.index if r.startswith("rule:")]
    best_rule = max(rules, key=lambda r: float(t.loc[r, "accuracy"]))
    gain = float(b["accuracy"]) - float(t.loc[best_rule, "accuracy"])
    lo, hi = ctx["ci"][best]["acc"]
    tn, fp, fn, tp = ctx["cm"].ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")

    if gain > 0.02:
        verdict = (f"The model beats the best hand-written rule ({best_rule}) by "
                   f"{gain * 100:.1f} percentage points.")
    elif gain > -0.02:
        verdict = (f"The model is about as good as the best hand-written rule ({best_rule}); "
                   f"the difference ({gain * 100:+.1f} points) is within noise.")
    else:
        verdict = (f"The model does NOT beat the best hand-written rule ({best_rule}), which is "
                   f"{-gain * 100:.1f} points better.")

    L = []
    L.append("# Drowsiness classifier — performance report\n")
    L.append(f"_Generated {datetime.datetime.now():%Y-%m-%d %H:%M}_\n")
    L.append("## Summary\n")
    L.append(f"- Task: tell **alert** from **drowsy** in {ctx['win']:.0f}-second windows of face-tracking features.")
    L.append(f"- Decision threshold: **{ctx.get('threshold', 0.5):g}** — a window is called "
             "drowsy when the model's probability reaches this. Every accuracy, recall and "
             "precision figure below is measured at it (ROC-AUC does not depend on it).")
    L.append(f"- Best model: **{best}** — accuracy **{b['accuracy']:.3f}** "
             f"(95% interval {lo:.3f}–{hi:.3f}), balanced accuracy {b['balanced_acc']:.3f}, "
             f"ROC-AUC {b['roc_auc']:.3f}, on people the model never saw during training.")
    L.append(f"- Chance level is {t.loc['coin_flip (majority class)', 'accuracy']:.3f}. {verdict}")
    L.append(f"- It catches {sens * 100:.0f}% of drowsy windows (sensitivity) and correctly clears "
             f"{spec * 100:.0f}% of alert windows (specificity).\n")

    L.append("## Data\n")
    L.append(f"- {ctx['n_windows']} windows from {ctx['n_videos']} videos and {ctx['n_people']} people "
             f"(alert: {ctx['n_alert']}, drowsy: {ctx['n_drowsy']}).")
    L.append(f"- Window settings: {ctx['window_params'] or 'not recorded'}")
    L.append(f"- Features used: {len(ctx['feature_cols'])} (feature set: `{ctx['feature_set']}`"
             + ("; baseline-dependent head-pose features INCLUDED" if ctx["used_baseline"]
                else "; baseline-dependent head-pose features excluded") + ").\n")

    L.append("## How it was tested\n")
    L.append(f"{ctx['n_splits']}-fold cross-validation **split by person**: every window is scored by a model "
             "trained only on other people, so the numbers reflect performance on new faces. "
             "Two kinds of reference points are included: a coin flip, and the simple hand-written rules "
             "from Sprints 1–2 (no learning).\n")

    L.append("## Results (window level, out-of-fold)\n")
    L.append(md_table(t[["accuracy", "balanced_acc", "precision", "recall", "f1", "roc_auc"]].astype(float)) + "\n")

    L.append("### 95% intervals (bootstrap over people)\n")
    ci_rows = {name: {"accuracy low": v["acc"][0], "accuracy high": v["acc"][1],
                      "AUC low": v["auc"][0], "AUC high": v["auc"][1]}
               for name, v in ctx["ci"].items()}
    L.append(md_table(pd.DataFrame(ci_rows).T) + "\n")
    L.append("Wide intervals mean the result could shift noticeably with a different set of people. "
             "If two rows' intervals overlap heavily, don't claim one is better.\n")

    L.append("### Accuracy on each held-out group of people\n")
    fa = pd.DataFrame({n: [f"{a:.3f}" for a in accs] for n, accs in ctx["fold_accs"].items()},
                      index=[f"fold {i + 1}" for i in range(ctx["n_splits"])])
    L.append(md_table(fa, "held-out group") + "\n")

    L.append("### Whole-video accuracy\n")
    L.append("Window probabilities averaged per video (each video has one label, so this is the fairest "
             "view of what the label means, and usually higher than window level):\n")
    vr = pd.DataFrame({"video accuracy": ctx["video_results"]})
    L.append(md_table(vr) + "\n")

    L.append(f"## Best model: {best}\n")
    L.append(f"| | predicted alert | predicted drowsy |\n|---|---|---|\n"
             f"| **actually alert** | {tn} | {fp} |\n| **actually drowsy** | {fn} | {tp} |\n")
    L.append("![confusion matrix](confusion_matrix.png)\n")
    L.append("![ROC curves](roc_curves.png)\n")

    L.append("## Which features mattered\n")
    imp = ctx["importances"].head(10).to_frame("importance")
    L.append(md_table(imp, "feature") + "\n")
    L.append("![feature importance](feature_importance.png)\n")
    L.append("Importance shows what the forest *used*, not that a feature *causes* drowsiness; "
             "correlated features share credit.\n")

    L.append("## Limitations (say these in the write-up)\n")
    L.append("- **Labels are per video and self-reported.** Every window in a \"drowsy\" video is labelled "
             "drowsy even if the person looked fine for a minute, which caps achievable accuracy.")
    L.append("- **Only clear alert vs. clear drowsy.** The in-between (\"low vigilant\") class was excluded, so "
             "nothing here shows the model catches *early* drowsiness.")
    L.append("- **Windows overlap by design**, so neighbouring windows are correlated; the people-wise split "
             "prevents leakage, but the effective sample size is closer to the number of people than windows.")
    L.append("- **Model choice uses the same folds it is scored on**, so the best-model number is slightly optimistic. "
             "Nothing was hyper-parameter tuned.")
    L.append("- **Dataset ≠ your live webcam.** Different cameras, angles and lighting. Person-relative eye "
             "features need a running baseline in the live system, and the live demo should be checked separately.")
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Train + evaluate alert-vs-drowsy models")
    ap.add_argument("--windows", default="windows.csv")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--splits", type=int, default=5, help="cross-validation folds (people-wise)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="probability above which a window is called drowsy. 0.5 is the "
                         "neutral default; diagnostics.py suggests a value tuned for your "
                         "recall / false-alarm trade-off. Saved into the model bundle so "
                         "the live system uses the same number.")
    ap.add_argument("--features", choices=["all", "raw", "normalized"], default="all",
                    help="eye features to use: all (default), raw (fixed-threshold only), "
                         "normalized (person-relative only). Run twice to compare.")
    ap.add_argument("--use-baseline-features", action="store_true",
                    help="include head-pose features measured against the first-3-seconds baseline")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.windows, dtype={"fold": str, "subject": str, "group": str})
    feature_cols = [c for c in df.columns if c not in META_COLS]
    if not args.use_baseline_features:
        feature_cols = [c for c in feature_cols if c not in BASELINE_DEPENDENT]
    if args.features == "raw":
        feature_cols = [c for c in feature_cols if c not in REL_FEATURE_NAMES]
    elif args.features == "normalized":
        feature_cols = [c for c in feature_cols if c not in RAW_EYE_FEATURES]
    X = df[feature_cols]
    y = df["label"].to_numpy(dtype=int)
    groups = df["group"].to_numpy()

    n_people = len(np.unique(groups))
    if n_people < 2:
        raise SystemExit("Need windows from at least 2 different people.")
    n_splits = min(args.splits, n_people)

    print(f"{len(df)} windows | {n_people} people | {len(feature_cols)} features")
    print("Windows per class:", {CLASS_NAMES[k]: int((y == k).sum()) for k in (0, 1)})
    print(f"Evaluation: {n_splits}-fold cross-validation, split by person")
    print(f"Decision threshold: {args.threshold:g}"
          + ("  (neutral default)" if args.threshold == 0.5 else "  (tuned)") + "\n")

    results = {}
    roc_data = {}

    # --- baselines (no learning) -----------------------------------------
    results["coin_flip (majority class)"] = {
        **score(y, np.full(len(y), float(np.mean(y)))),
        "accuracy": max(np.mean(y), 1 - np.mean(y)),
    }
    results["coin_flip (majority class)"]["roc_auc"] = 0.5

    long_closure = df["closure_max_dur"].fillna(0).to_numpy()     # Sprint 1 idea: eyes shut ~0.5 s
    perclos = df["closed_frac"].fillna(0).to_numpy()               # Sprint 2 idea: PERCLOS > 30%
    rule1 = score(y, (long_closure >= 0.5).astype(float))
    rule1["roc_auc"] = roc_auc_score(y, long_closure)
    rule2 = score(y, (perclos > 0.30).astype(float))
    rule2["roc_auc"] = roc_auc_score(y, perclos)
    results["rule: closure >= 0.5s (Sprint 1)"] = rule1
    results["rule: PERCLOS > 30% (Sprint 2)"] = rule2
    roc_data["rule: closure >= 0.5s"] = roc_curve(y, long_closure)[:2]
    roc_data["rule: PERCLOS"] = roc_curve(y, perclos)[:2]

    # --- models, person-wise cross-validation -----------------------------
    cv = GroupKFold(n_splits=n_splits)
    oof_probs = {}
    video_results = {}
    fold_accs = {}
    for name, model in make_models(args.seed).items():
        oof = np.zeros(len(y))
        fold_acc = []
        for tr, te in cv.split(X, y, groups):
            m = clone(model).fit(X.iloc[tr], y[tr])
            oof[te] = m.predict_proba(X.iloc[te])[:, 1]
            fold_acc.append(accuracy_score(y[te], oof[te] >= args.threshold))
        oof_probs[name] = oof
        fold_accs[name] = fold_acc
        res = score(y, oof, args.threshold)
        res["acc_across_folds"] = f"{np.mean(fold_acc):.3f} ± {np.std(fold_acc):.3f}"
        results[name] = res
        roc_data[name] = roc_curve(y, oof)[:2]

        # whole-video view: average the window probabilities per video
        vid = pd.DataFrame({"video_id": df["video_id"], "y": y, "p": oof}).groupby("video_id").agg(
            y=("y", "first"), p=("p", "mean"))
        video_results[name] = accuracy_score(vid["y"], vid["p"] >= args.threshold)
        print(f"  finished {name}")

    table = pd.DataFrame(results).T
    table.index.name = "model"
    table.to_csv(os.path.join(args.out_dir, "cv_results.csv"))
    print("\n=== Window-level results (out-of-fold, unseen people) ===")
    show = table[["accuracy", "balanced_acc", "precision", "recall", "f1", "roc_auc"]].astype(float)
    print(show.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nAccuracy spread across folds (mean ± std):")
    for name in make_models(args.seed):
        print(f"  {name}: {results[name]['acc_across_folds']}")
    print("\nWhole-video accuracy (average window probability per video; easier than window-level):")
    for name, acc in video_results.items():
        print(f"  {name}: {acc:.3f}")

    # --- pick the best model by ROC-AUC, plot, refit on everything ---------
    model_names = list(make_models(args.seed))
    best = max(model_names, key=lambda n: results[n]["roc_auc"])
    print(f"\nBest model by ROC-AUC: {best}")

    cm = confusion_matrix(y, (oof_probs[best] >= args.threshold).astype(int))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{best}\n(unseen people, threshold {args.threshold:g})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (fpr, tpr) in roc_data.items():
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (unseen people)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "roc_curves.png"), dpi=150)
    plt.close(fig)

    # feature importance from a forest trained on all the data
    rf = make_models(args.seed)["random_forest"].fit(X, y)
    importances = pd.Series(rf.named_steps["rf"].feature_importances_,
                            index=feature_cols).sort_values(ascending=False)
    print("\nTop features (random forest, trained on all data):")
    print(importances.head(10).to_string(float_format=lambda v: f"{v:.3f}"))
    top = importances.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(top.index, top.values)
    ax.set_title("Feature importance (random forest)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "feature_importance.png"), dpi=150)
    plt.close(fig)

    final = clone(make_models(args.seed)[best]).fit(X, y)
    window_params = {}
    meta_path = args.windows + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            window_params = json.load(f)
    joblib.dump({"model": final, "feature_names": feature_cols, "classes": CLASS_NAMES,
                 "threshold": args.threshold, "window_params": window_params,
                 "model_name": best},
                os.path.join(args.out_dir, "drowsiness_model.joblib"))
    # --- written report ---------------------------------------------------
    ci = {n: dict(zip(("acc", "auc"), bootstrap_ci(y, oof_probs[n], groups, seed=args.seed,
                                                   threshold=args.threshold)))
          for n in model_names}
    ci["rule: closure >= 0.5s (Sprint 1)"] = dict(zip(("acc", "auc"), bootstrap_ci(
        y, (long_closure >= 0.5).astype(float), groups, seed=args.seed)))
    ci["rule: PERCLOS > 30% (Sprint 2)"] = dict(zip(("acc", "auc"), bootstrap_ci(
        y, (perclos > 0.30).astype(float), groups, seed=args.seed)))
    write_report(os.path.join(args.out_dir, "report.md"), {
        "table": table, "best": best, "cm": cm, "ci": ci, "fold_accs": fold_accs,
        "video_results": video_results, "importances": importances,
        "n_windows": len(df), "n_videos": df["video_id"].nunique(), "n_people": n_people,
        "n_alert": int((y == 0).sum()), "n_drowsy": int((y == 1).sum()),
        "n_splits": n_splits, "win": float(window_params.get("win_seconds", 30.0)),
        "window_params": window_params, "feature_cols": feature_cols,
        "feature_set": args.features, "used_baseline": args.use_baseline_features,
        "threshold": args.threshold,
    })
    print(f"\nSaved results, plots, model and report.md to {args.out_dir}/")


if __name__ == "__main__":
    main()
