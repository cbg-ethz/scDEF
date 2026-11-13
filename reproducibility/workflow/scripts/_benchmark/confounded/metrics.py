"""The two pre-registered metrics.

Metric 1 (calibration): per cell type with >= 60 cells in each batch, r_c = cross-validated batch
AUC within the type on a method's embedding; t_c = the same on the uncorrected PCA of the clean
data (0.5 where there is no biology by construction). Primary: cell-weighted mean |r_c - t_c|.

Metric 2 (attribution): area under the precision-recall curve of each method's per-type gene
scores against the true response genes (cell-weighted over types with a non-empty truth), and of
its technical score against the true technical genes where those are defined.
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MIN_CELLS_PER_BATCH = 60


def batch_auc(emb, batch, seed=0):
    """3-fold stratified, standardised-in-fold logistic regression; out-of-fold AUC."""
    y = (batch == np.unique(batch)[1]).astype(int)
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=seed).split(emb, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(emb[tr], y[tr])
        oof[te] = clf.predict_proba(emb[te])[:, 1]
    return float(roc_auc_score(y, oof))


def eligible_types(ctype, batch):
    out = []
    for t in np.unique(ctype):
        m = ctype == t
        if all((batch[m] == b).sum() >= MIN_CELLS_PER_BATCH for b in np.unique(batch)):
            out.append(t)
    return out


def per_type_auc(emb, ctype, batch, types):
    return {t: batch_auc(emb[ctype == t], batch[ctype == t]) for t in types}


def calibration(r, t, weights):
    types = [k for k in t if k in r]
    w = np.array([weights[k] for k in types], dtype=float)
    d = np.array([abs(r[k] - t[k]) for k in types])
    rho = spearmanr([r[k] for k in types], [t[k] for k in types]).statistic \
        if len(types) > 2 and np.std([t[k] for k in types]) > 0 else float("nan")
    return dict(mae=float((d * w).sum() / w.sum()), spearman=float(rho),
                per_type={k: dict(r=r[k], t=t[k]) for k in types})


def attribution(bio_scores, tech_score, var_names, bio_truth, tech_truth, weights):
    var_names = np.asarray(var_names)
    res = {}
    if bio_scores is not None:
        aps, ws, per = [], [], {}
        for t, genes in bio_truth.items():
            if not genes or t not in weights:
                continue
            y = np.isin(var_names, genes).astype(int)
            s = bio_scores.get(t, np.zeros(len(var_names)))
            ap = float(average_precision_score(y, s)) if s.any() else float(y.mean())
            per[t] = dict(ap=ap, prevalence=float(y.mean()))
            aps.append(ap)
            ws.append(weights[t])
        if aps:
            res["bio_auprc"] = float(np.average(aps, weights=ws))
            res["bio_per_type"] = per
    if tech_score is not None and tech_truth:
        y = np.isin(var_names, tech_truth).astype(int)
        res["tech_auprc"] = float(average_precision_score(y, tech_score))
        res["tech_prevalence"] = float(y.mean())
    return res
