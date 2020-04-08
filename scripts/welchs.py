from typing import List

import numpy as np
from scipy import stats
from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)


class Args(Tap):
    mean1: List[float]  # Means of distributions of 1st model
    mean2: List[float]  # Means of distributions of 2nd model
    std1: List[float]  # Standard deviations of distributions of 1st model
    std2: List[float]  # Standard deviations of distributions of 2nd model
    nobs1: List[int]  # Number of observations each mean/std is constructed from in 1st model
    nobs2: List[int]  # Number of observations each mean/std is constructed from in 2nd model


def welchs(mean1: List[float],  # mean performance across folds for each dataset (model 1)
           std1: List[float],  # standard deviation performance across folds for each dataset (model 1)
           nobs1: List[int],  # number of CV folds for each dataset (model 1)
           mean2: List[float],  # mean performance across folds for each dataset (model 2)
           std2: List[float],  # standard deviation performance across folds for each dataset (model 2)
           nobs2: List[int]):  # number of CV folds for each dataset (model 2)
    # Expand one number of observations to all
    if len(nobs1) == 1:
        nobs1 = nobs1 * len(mean1)

    if len(nobs2) == 1:
        nobs2 = nobs2 * len(mean2)

    assert len(mean1) == len(std1) == len(nobs1) == len(mean2) == len(std2) == len(nobs2)

    # Convert from population standard deviation to sample standard deviation
    std1 = [s * np.sqrt(n / (n - 1)) for s, n in zip(std1, nobs1)]
    std2 = [s * np.sqrt(n / (n - 1)) for s, n in zip(std2, nobs2)]

    # Compute Welch's t-test p-values for each dataset based on mean, standard deviation, and number of observations
    pvalues = [
        stats.ttest_ind_from_stats(mean1=m1, std1=s1, nobs1=n1, mean2=m2, std2=s2, nobs2=n2, equal_var=False).pvalue / 2
        for m1, s1, n1, m2, s2, n2 in zip(mean1, std1, nobs1, mean2, std2, nobs2)
    ]

    # Print Welch's p-values
    print('\n'.join(f'{pvalue:.4e}' for pvalue in pvalues))

    # Chi-squared statistic
    chisquare = -2 * np.sum(np.log(pvalues))
    print(f'X^2  = {chisquare}')

    # Degrees of freedom
    df = 2 * len(pvalues)
    print(f'df = {df}')

    # Two-sided p-value for chi-squared
    pvalue = 1 - stats.distributions.chi2.cdf(chisquare, df=df)

    # Print p-value
    print(f'p = {pvalue:.4e}')


if __name__ == '__main__':
    args = Args().parse_args()

    welchs(
        mean1=args.mean1,
        std1=args.std1,
        nobs1=args.nobs1,
        mean2=args.mean2,
        std2=args.std2,
        nobs2=args.nobs2
    )
