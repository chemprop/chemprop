import pandas as pd
import numpy as np

alpha_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

targets = pd.read_csv("data/tox21_multilabel3.csv")
targets = targets.replace(np.nan, 0.5)

f = open("multilabel_new/coverage.csv", "w")
f.write("alpha,")
for task in targets.columns[1:]:
    f.write(f"{task}_coverage,")
f.write("coverage\n")

for alpha in alpha_list:
    evals = pd.read_csv(f"multilabel/tox21_eval_conformal_multilabel_{alpha}.csv")
    f.write(f"{alpha},")
    for task in targets.columns[1:]:
        f.write(f"{evals[task][0]},")
    f.write("\n")

f.close()

"""

for alpha in alpha_list:
    preds = pd.read_csv(f"multilabel_new/tox21_preds_conformal_multilabel_{alpha}.csv")

    results = np.full(len(targets.index), True)
    coverage_id = {}

    for task in targets.columns[1:]:
        task_targets = targets[task].to_numpy()
        in_set = preds[task+"_conformal_in_set"].to_numpy()
        out_set = preds[task+"_conformal_out_set"].to_numpy()

        task_results = np.logical_and(in_set <= task_targets, task_targets <= out_set)
        results = np.logical_and(results, task_results)
        coverage_id[task] = task_results.sum() / task_results.shape[0]
        # There shouldn't be any per task coverage guarantee as the bound is over all tasks together.

    print(results)
    coverage = results.sum() / results.shape[0]

    f.write(f"{alpha},")
    for task in targets.columns[1:]:
        f.write(f"{coverage_id[task]},")
    f.write(f"{coverage}\n")

f.close()

"""