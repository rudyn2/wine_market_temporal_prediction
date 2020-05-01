import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import f, t
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison


observations_as_dict = {
    '15%': [7, 7, 15, 11, 9],
    '20%': [12, 17, 12, 18, 18],
    '25%': [14, 18, 18, 19, 19],
    '30%': [19, 25, 22, 19, 23],
    '35%': [7, 10, 11, 15, 11]
}
cotton_percentages = []
values = []
for key, value in observations_as_dict.items():
    for _ in range(len(value)):
        cotton_percentages.append(key)
    values = values + value

observations_df = pd.DataFrame(data=observations_as_dict)
flatten_observations_df = pd.DataFrame({'cotton_percentage': cotton_percentages,
                                        'tensile_strength': values})
flatten_observations_means = flatten_observations_df.groupby('cotton_percentage').mean()

book_example = np.array([[7, 8, 15, 11, 9, 10],
                         [12, 17, 13, 18, 19, 15],
                         [14, 18, 19, 17, 16, 18],
                         [19, 25, 22, 23, 18, 20]])


def calculate_anova_Fo(data_as_array: np.ndarray):
    a = data_as_array.shape[0]
    n = data_as_array.shape[1]
    N = a*n
    ydotdot = data_as_array.sum()
    SSt = (data_as_array**2).sum() - ydotdot**2/N
    SStreatments = (data_as_array.sum(axis=1)**2).sum() / n - ydotdot**2 / N
    SSe = SSt - SStreatments
    MStreatment = SStreatments/(a-1)
    MSe = SSe/(a*(n-1))
    print(f"Fo = {(MStreatment/MSe):.5f}")
    return SSt, SStreatments, SSe, MStreatment, MSe


if __name__ == '__main__':
    # step 1: explore distribution
    box_plot = observations_df.boxplot()
    box_plot.set(xlabel='Treatment', ylabel='Tensile strength')
    # plt.show()

    # step 2: reject ANOVA null hypothesis
    St, SStreatments, SSe, MSt, MSe = calculate_anova_Fo(observations_df.values)
    print("\n")

    # step 3: mean multiple comparison by Fisher method or pairwise comparison (Tukey HSD)
    mc = MultiComparison(flatten_observations_df['tensile_strength'], flatten_observations_df['cotton_percentage'])
    result = mc.tukeyhsd()
    print(result)
    print(mc.groupsunique)

    # step 4: identify groups
    # step 5 (optional?): check ANOVA assumption 1: normal distribution of residuals using Shapiro-Wilk test
    # step 6 (optional?): check ANOVA assumption 2: Homogeneity of variances using Levene or Bartlett test
    # step 7: check ANOVA hypothesis on each group
