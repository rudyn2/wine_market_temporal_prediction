import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import f, t
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import scipy.stats as stats


cotton_tensile_strength = pd.DataFrame({
    '15%': [7, 7, 15, 11, 9],
    '20%': [12, 17, 12, 18, 18],
    '25%': [14, 18, 18, 19, 19],
    '30%': [19, 25, 22, 19, 23],
    '35%': [7, 10, 11, 15, 11]
})

book_example = pd.DataFrame({'5%': [7, 8, 15, 11, 9, 10],
                             '10%': [12, 17, 13, 18, 19, 15],
                             '15%': [14, 18, 19, 17, 16, 18],
                             '20%': [19, 25, 22, 23, 18, 20]})


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

    working_data = cotton_tensile_strength

    # step 1: explore distribution
    box_plot = working_data.boxplot()
    box_plot.set(xlabel='Treatment', ylabel='Tensile strength')
    plt.show()

    # reshape the d dataframe suitable for statsmodels package
    d_melt = pd.melt(working_data.reset_index(), id_vars=['index'], value_vars=working_data.columns.values)
    # replace column names
    d_melt.columns = ['index', 'treatments', 'value']
    # Ordinary Least Squares (OLS) model
    model = ols('value ~ C(treatments)', data=d_melt).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    print("\n")

    # step 3: mean multiple comparison by Fisher method or pairwise comparison (Tukey HSD)
    mc = MultiComparison(d_melt['value'], d_melt['treatments'])
    result = mc.tukeyhsd()
    print(result)
    print(mc.groupsunique)

    # step 4: identify groups
    # step 5 (optional?): check ANOVA assumption 1: normal distribution of residuals using Shapiro-Wilk test
    w, pvalue = stats.shapiro(model.resid)
    print(w, pvalue)

    # step 6 (optional?): check ANOVA assumption 2: Homogeneity of variances using Levene or Bartlett test
    # step 7: check ANOVA hypothesis on each group
