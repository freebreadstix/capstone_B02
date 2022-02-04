import pandas as pd

def get_Xy_from_sheet(file):
    df = pd.read_csv(file)
    simple_df = df[["Original article text", "Verdict"]]
    X = simple_df["Original article text"].values
    y = simple_df["Verdict"].replace("FALSE",1).replace("TRUE",0).values
    return X, y