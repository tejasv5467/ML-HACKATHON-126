"""PathSnap five-model training; consumes development feature caches.
Adapted from the existing PathSnap project; preparation is supplied by the EDA owner.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, confusion_matrix
import joblib
from training_cache import validate_development

FEATURES = ['distance_m', 'bearing_difference_deg', 'speed_mps', 'arc_length_m', 'projection_fraction', 'time_gap_s', 'candidate_rank']
K = 5
SEED = 42


# Preparation is intentionally a separate team work package.
def _preparation_module():
    try:
        import pathsnap_real_preprocessing as prep
    except ModuleNotFoundError as exc:
        if exc.name != 'pathsnap_real_preprocessing':
            raise
        raise RuntimeError('Integrate the EDA member\'s pathsnap_real_preprocessing.py first.') from exc
    return prep


def read_table(*args, **kwargs):
    return _preparation_module().read_table(*args, **kwargs)


def candidate_features(*args, **kwargs):
    return _preparation_module().candidate_features(*args, **kwargs)


def prepare_trip(*args, **kwargs):
    return _preparation_module().prepare_trip(*args, **kwargs)


def prepare(root):
    return _preparation_module().prepare(root)


def models():
    return {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', max_iter=500, random_state=SEED)),
        'Gaussian Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(max_depth=12, min_samples_leaf=10, random_state=SEED),
        'Random Forest': RandomForestClassifier(n_estimators=40, max_depth=12, min_samples_leaf=10, n_jobs=2, random_state=SEED),
        'Histogram Gradient Boosting': HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=31, early_stopping=False, random_state=SEED),
    }


def train(root):
    cache = root / 'data' / 'processed'
    out = root / 'results' / 'official_baseline'; out.mkdir(parents=True, exist_ok=True)
    artifacts = root / 'artifacts' / 'official_baseline'; artifacts.mkdir(parents=True, exist_ok=True)
    with np.load(cache / 'train.npz') as a:
        X, y, group = a['X'], a['y'], a['group']
    with np.load(cache / 'student_test.npz') as a:
        tx, ty, tg, tf, tt, ti = (a[key] for key in ('X', 'y', 'group', 'from_node', 'to_node', 'ids'))
    validate_development(X, y, group, tx, ty, tg, ti, tf, tt)
    xf, yf = X.reshape(-1, len(FEATURES)), y.ravel()
    xt, yt = tx.reshape(-1, len(FEATURES)), ty.ravel()
    folds = list(GroupKFold(n_splits=5).split(X, groups=group))
    rows, fold_rows = [], []
    best = -1.0
    best_name = None
    configurations = {}
    for name, model in models().items():
        print(f'\nTraining {name}', flush=True)
        configurations[name] = {k: str(v) for k, v in model.get_params().items()}
        scores = []
        for f, (tr, va) in enumerate(folds, 1):
            m = clone(model).fit(X[tr].reshape(-1, len(FEATURES)), y[tr].ravel())
            pred = m.predict(X[va].reshape(-1, len(FEATURES)))
            score = f1_score(y[va].ravel(), pred, average='macro', zero_division=0)
            scores.append(score)
            fold_rows.append({'model': name, 'fold': f, 'macro_f1': score})
            print(f'  Fold {f}/5: Macro F1 {score:.4f}', flush=True)
        tick = time.perf_counter()
        fitted = clone(model).fit(xf, yf)
        train_sec = time.perf_counter() - tick
        tick = time.perf_counter()
        prob = fitted.predict_proba(xt)
        inference_sec = time.perf_counter() - tick
        if not np.isfinite(prob).all() or not np.allclose(prob.sum(axis=1), 1):
            raise ValueError(f'Invalid probabilities for {name}')
        pos = int(np.flatnonzero(fitted.classes_ == 1)[0])
        raw = fitted.classes_[np.argmax(prob, axis=1)]
        top = prob[:, pos].reshape(-1, K).argmax(axis=1)
        ix = np.arange(len(tx))
        chosen = ti[ix, top]
        chosen_from, chosen_to = tf[ix, top], tt[ix, top]
        within = tg[1:] == tg[:-1]
        contiguous = (chosen[1:] == chosen[:-1]) | (chosen_from[1:] == chosen_to[:-1])
        row = dict(model=name, cv_macro_f1=np.mean(scores), cv_f1_std=np.std(scores),
                   candidate_accuracy=accuracy_score(yt, raw), student_macro_f1=f1_score(yt, raw, average='macro', zero_division=0),
                   precision=precision_score(yt, raw, zero_division=0), recall=recall_score(yt, raw, zero_division=0),
                   top1_proxy_accuracy=np.mean(ty[ix, top]), candidate_miss_rate=np.mean(~ty.any(axis=1)),
                   candidate_log_loss=log_loss(yt, prob, labels=fitted.classes_),
                   direct_route_continuity=np.mean(contiguous[within]), training_seconds=train_sec, inference_seconds=inference_sec)
        rows.append(row)
        slug = name.lower().replace(' ', '_')
        joblib.dump(fitted, artifacts / (slug + '.joblib'))
        np.savetxt(out / (slug + '_confusion.csv'), confusion_matrix(yt, raw, labels=[0, 1]), fmt='%d', delimiter=',')
        if row['cv_macro_f1'] > best:
            best, best_name = row['cv_macro_f1'], name
            joblib.dump(fitted, artifacts / 'best_untuned.joblib')
        pd.DataFrame(rows).to_csv(out / 'model_comparison.csv', index=False)
        pd.DataFrame(fold_rows).to_csv(out / 'cv_folds.csv', index=False)
        print(row, flush=True)
    (artifacts / 'configuration.json').write_text(json.dumps({'features': FEATURES, 'k': K, 'seed': SEED,
        'best_by_cv_macro_f1': best_name, 'label_method': 'nearest reference segment proxy; no sequence alignment', 'models': configurations}, indent=2))
    df = pd.DataFrame(rows).sort_values('cv_macro_f1', ascending=False)
    print('\nDEVELOPMENT BASELINE: PROXY LABELS', flush=True)
    print(df.round(4).to_string(index=False), flush=True)
    print('Best by training-trip CV:', best_name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric in ('cv_macro_f1', 'student_macro_f1', 'top1_proxy_accuracy', 'candidate_log_loss', 'direct_route_continuity'):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.barh(df.model, df[metric]); ax.invert_yaxis(); ax.set_xlabel(metric); ax.set_title('Development baseline — proxy labels'); fig.tight_layout()
        fig.savefig(out / (metric + '.png'), dpi=160); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'train'])
    p.add_argument('--root', type=Path, default=Path.cwd())
    args = p.parse_args()
    (prepare if args.stage == 'prepare' else train)(args.root.resolve())
