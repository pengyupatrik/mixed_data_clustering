import pandas as pd
from sklearn.preprocessing import StandardScaler


def transform(df, scaler=None, dummy=False):
    df = df.copy()
    df[['Ever_Married', 'Graduated', 'Profession']] = df[['Ever_Married',
                                                          'Graduated', 'Profession']].fillna('Unknown')
    df['Work_Experience'].fillna(1, inplace=True)
    df['Family_Size'].fillna(2, inplace=True)
    df.drop('Var_1', axis=1, inplace=True)
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(df[['Age', 'Work_Experience', 'Family_Size']])

    df[['Age', 'Work_Experience', 'Family_Size']] = scaler.transform(df[['Age',
                                                                         'Work_Experience', 'Family_Size']])
    if dummy is True:
        cat_cols = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score']
        dummies = pd.get_dummies(df[cat_cols], drop_first=True)
        return pd.concat([df.drop(cat_cols, axis=1), dummies], axis=1)
    return df


if __name__ == '__main__':
    df = pd.read_csv('data/Train.csv')
    scaler = StandardScaler()
    scaler.fit(df[['Age', 'Work_Experience', 'Family_Size']])
    transform(df, scaler).to_csv('data/train_no_missing_standard.csv', index=False)
    transform(df, scaler, dummy=True).to_csv('data/train_no_missing_standard_binary.csv', index=False)
