import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison

SMALL_SIZE = 16
MEDIUM_SIZE = 16
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

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
    N = a * n
    ydotdot = data_as_array.sum()
    SSt = (data_as_array ** 2).sum() - ydotdot ** 2 / N
    SStreatments = (data_as_array.sum(axis=1) ** 2).sum() / n - ydotdot ** 2 / N
    SSe = SSt - SStreatments
    MStreatment = SStreatments / (a - 1)
    MSe = SSe / (a * (n - 1))
    print(f"Fo = {(MStreatment / MSe):.5f}")
    return SSt, SStreatments, SSe, MStreatment, MSe


if __name__ == '__main__':
    working_data = cotton_tensile_strength

    # step 1: explore distribution
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    box_plot = working_data.boxplot(ax=ax)
    box_plot.set(xlabel='Porcentaje de algodón', ylabel='Resistencia a la tensión')
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

    # step 5: check ANOVA assumption 1: normal distribution of residuals using Shapiro-Wilk test
    w, pvalue = stats.shapiro(model.resid)
    print(w, pvalue)
