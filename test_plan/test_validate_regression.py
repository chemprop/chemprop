import pandas as pd
import numpy as np
from pandas import DataFrame as df

alpha_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.45]

targets = pd.read_csv("data/delaney3class3.csv")

f = open("regression/coverage.csv", "w")
f.write("alpha,")
for task in targets.columns[1:]:
    f.write(f"{task}_coverage,")
f.write("coverage\n")

for alpha in alpha_list:
    evals = pd.read_csv(f"regression/delaney_eval_conformal_{alpha}.csv")
    f.write(f"{alpha},")
    for task in targets.columns[1:]:
        f.write(f"{evals[task][0]},")
    f.write("\n")

f.close()

"""

for alpha in alpha_list:
    preds = pd.read_csv(f"regression/delaney_preds_conformal_{alpha}.csv")

    results = np.full(len(targets.index), True)
    coverage_id = {}

    for task in targets.columns[1:]:
        task_targets = targets[task].to_numpy()
        lower_bound = preds[task+"_lower_bound"].to_numpy()
        upper_bound = preds[task+"_upper_bound"].to_numpy()

        task_results = np.logical_and(lower_bound <= task_targets, task_targets <= upper_bound)
        results = np.logical_and(results, task_results)
        coverage_id[task] = task_results.sum() / task_results.shape[0]

    print(results)
    coverage = results.sum() / results.shape[0]

    f.write(f"{alpha},")
    for task in targets.columns[1:]:
        f.write(f"{coverage_id[task]},")
    f.write(f"{coverage}\n")

f.close()

"""