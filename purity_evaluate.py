import pandas as pd


def purity(df, input_type='raw', true_col='true', pred_col='pred'):
    if input_type == 'cross_tab':
        pdf = df.copy()
    else:
        pdf = pd.crosstab(df[pred_col], df[true_col])
    pj = pdf.div(pdf.sum(axis=1), axis=0).max(axis=1)
    mj = pdf.sum(axis=1)/pdf.sum(axis=1).sum()
    sum_ = 0
    for j in pj.index:
        sum_ += pj[j] * mj[j]
    return sum_


if __name__ == '__main__':
    pdf = pd.read_csv('data/purity_test.csv')
    pdf.set_index('cluster', inplace=True)
    assert round(purity(pdf, input_type='cross_tab'), 4) == 0.7203
