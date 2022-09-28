import pandas as pd
import numpy as np
from pandas import DataFrame as df

alpha_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

targets = pd.read_csv("data/tox21_3class3.csv")

f = open("multiclass/coverage.csv", "w")
f.write("alpha,")
for task in targets.columns[1:]:
    f.write(f"{task}_coverage,")
f.write("coverage\n")

for alpha in alpha_list:
    preds = pd.read_csv(f"multiclass/tox21_preds_conformal_multiclass_{alpha}.csv")

    results = np.full(len(targets.index), True)
    coverage_id = {}

    for task in targets.columns[1:]:
        task_results = np.full(len(targets.index), False)
        for i in targets.index:
            true_class = targets[task][i]
            task_results[i] = bool(preds[f"{task}_class_{true_class}_conformal"][i])

        results = np.logical_and(results, task_results)
        coverage_id[task] = task_results.sum() / task_results.shape[0]

    print(results)
    coverage = results.sum() / results.shape[0]

    f.write(f"{alpha},")
    for task in targets.columns[1:]:
        f.write(f"{coverage_id[task]},")
    f.write(f"{coverage}\n")

f.close()