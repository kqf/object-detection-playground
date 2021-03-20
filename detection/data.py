import pandas as pd


def read_data(path):
    df = pd.read_csv(path)
    df.fillna(0, inplace=True)
    df.loc[df["class_id"] == 14, ['x_max', 'y_max']] = 1.0

    df["class_id"] = df["class_id"] + 1
    df.loc[df["class_id"] == 15, ["class_id"]] = 0
    return df
