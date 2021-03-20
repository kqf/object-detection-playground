import pandas as pd


def read_data(path):
    df = pd.read_csv(path)
    df.loc[df["class_id"] == 14, ['x_max', 'y_max']] = 1000
    df.loc[df["class_id"] == 14, ['x_min', 'y_min']] = 0.001

    df["class_id"] = df["class_id"] + 1
    df.loc[df["class_id"] == 15, ["class_id"]] = 0
    return df
