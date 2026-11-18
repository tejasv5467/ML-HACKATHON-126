# Model training and Bayesian optimisation

This work package ports the existing development experiment from [PathSnap-ML-Hackathon at 8eca2a2152a6557870ae4e9a11bee8c92dcca6f3](https://github.com/tejasv5467/PathSnap-ML-Hackathon/tree/8eca2a2152a6557870ae4e9a11bee8c92dcca6f3). The recorded statistics are imported results, not a new training run on this repository. No reserved evaluator data are used.

## Integration contract

The EDA owner provides `pathsnap_real_preprocessing.py`, exporting `read_table`, `candidate_features`, `prepare_trip`, and `prepare`. The training module preserves the previous entry points by delegating preparation to that module. The default root is the working directory. The cache contains `data/processed/train.npz` and `student_test.npz`. Both partitions hold X (fixes × 5 × 7), y (fixes × 5, binary nearest-reference proxy), group (one trip identifier per fix), ids, from_node and to_node (fixes × 5). The feature order must match FEATURES in the training module. There are 80 train and 10 student-test trips with disjoint IDs. All-negative candidate groups are allowed when the reference arc is missing from the candidates.

## Reproduction

Install the declared requirements in an isolated Python environment. After integrating preprocessing and placing only the authorized development data in the expected directories, run:

```sh
python pathsnap_real_training.py prepare
python pathsnap_real_training.py train
python pathsnap_real_tune.py --root .
python -m unittest discover -s tests -p 'test_training*.py'
```

The trainer compares scaled logistic regression, Gaussian naive Bayes, decision tree, random forest and histogram gradient boosting. It uses five grouped folds so individual trips do not appear in both fit and validation splits. StandardScaler is inside the logistic pipeline. Training uses candidate-level binary labels. Selection uses mean training-fold macro F1; student-test metrics are reported after fitting. The frozen tuned model is saved at artifacts/official_tuning/tuned_model.joblib after a successful run, for the deployment owner. Model binaries are not included in this port; regenerate with the development caches or import verified existing artifacts separately.

The TPE search uses 15 trials, five startup trials and seed 42. It optimises max_iter, learning_rate, max_leaf_nodes, min_samples_leaf and l2_regularization on the training folds only. The full original trial history is not available; a new run writes its own bayesian_trials.csv. The original best parameters and actual recorded results are preserved without manufacturing new trials.

## Recorded results

| Model | Mean CV macro F1 |
| --- | ---: |
| Logistic regression | 0.83113 |
| Gaussian naive Bayes | 0.80109 |
| Decision tree | 0.85362 |
| Random forest | 0.85634 |
| Histogram gradient boosting | 0.85901 |
| Tuned histogram gradient boosting | 0.86000 |

HGB was chosen by the highest baseline mean CV macro F1, not by the best student-test score. The tuned candidate accuracy was 92.01%. Top-1 proxy accuracy was 80.56% before sequence decoding. These are development proxy metrics, not verified per-fix ground truth or reserved evaluator scores. The small CV gain is descriptive, not a claim of statistical significance. Evaluation graphs and HMM/deployment integration remain separate work packages.

## Review

The new tests use small synthetic inputs strictly as software fixtures and verify estimator probability outputs and trip isolation. They are not experimental metrics. Check prerequisites and run the full development pipeline before treating the imported scores as independently reproduced.
