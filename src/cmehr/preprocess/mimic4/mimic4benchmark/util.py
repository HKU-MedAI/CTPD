import pandas as pd

# mimic-iv tables don't have row_id as index column
def dataframe_from_csv(path, header=0, index_col=None):
    return pd.read_csv(path, header=header, index_col=index_col)
