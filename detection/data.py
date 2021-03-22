import pandas as pd


def read_data(path):
    df = pd.read_csv(path)
    df.loc[df["class_id"] == 14, 'x_min'] = 691.0
    df.loc[df["class_id"] == 14, 'x_max'] = 1653.0
    df.loc[df["class_id"] == 14, 'y_min'] = 1375.0
    df.loc[df["class_id"] == 14, 'y_max'] = 1831.0

    df["class_id"] = df["class_id"] + 1
    df.loc[df["class_id"] == 15, ["class_id"]] = 0
    return df
