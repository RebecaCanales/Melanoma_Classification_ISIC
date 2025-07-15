import pandas as pd

def save_dataframe(df, path):
    df.to_excel(path, index=False)

def load_dataframe(path):
    return pd.read_excel(path)
