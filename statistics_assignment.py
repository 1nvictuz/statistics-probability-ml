from pathlib import Path
import sys
import pandas as pd
from scipy.stats import ttest_ind
import statsmodels.api as sm
import scipy

folder = Path(__file__).resolve().parent
df = pd.read_csv(folder / 'dataset_1.csv')

print('STATISTICS AND PROBABILITY ASSIGNMENT')
print('Number of rows:', len(df))
print('Number of columns:', len(df.columns))
print('\nMissing values by column:\n', df.isna().sum().to_string())
print('\nDescriptive statistics:\n', df.describe(include='all').to_string())

probability = (df['target'] == 1).mean()
print(f'\nProbability of target = 1: {probability:.2%}')

# Two-sided Welch t-test: average column_1 for groups A and B
A = df.loc[df['group'] == 'A', 'column_1']
B = df.loc[df['group'] == 'B', 'column_1']
t_stat, p_value = ttest_ind(A, B, equal_var=False)
print(f'\nWelch t-test (column_1, A vs B): t={t_stat:.4f}, p={p_value:.6f}')

numeric = df.select_dtypes(include='number')
print('\nCorrelation matrix:\n', numeric.corr().to_string())

# Illustrative OLS: predict a continuous variable, feature_1.
X = sm.add_constant(df[['column_1', 'column_2']])
y = df['feature_1']
model = sm.OLS(y, X).fit()
print('\nOLS regression (feature_1 ~ column_1 + column_2):\n', model.summary())

results = [
    'Statistics and Probability for Machine Learning Assignment',
    f'Python: {sys.version.split()[0]}',
    f'pandas: {pd.__version__}',
    f'SciPy: {scipy.__version__}',
    f'statsmodels: {sm.__version__}',
    f'Rows: {len(df)}; columns: {len(df.columns)}',
    '\nMissing values:\n' + df.isna().sum().to_string(),
    '\nDescriptive statistics:\n' + df.describe(include='all').to_string(),
    f'\nProbability(target=1): {probability:.2%}',
    f'\nWelch t-test (column_1; A vs B): t={t_stat:.4f}; p={p_value:.6f}',
    '\nCorrelation matrix:\n' + numeric.corr().to_string(),
    '\nOLS summary:\n' + model.summary().as_text(),
    '\nCaution: This sample is small. Hypothesis-test assumptions, independence, and possible data selection need review; association is not causation.'
]
(folder / 'statistics_results.txt').write_text('\n'.join(results), encoding='utf-8')
print('\nResults saved to statistics_results.txt')
