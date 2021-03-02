import pandas as pd


def test_dummy(fake_dataset):
    df = pd.read_csv(fake_dataset / "train.csv")
    print(df.head())
