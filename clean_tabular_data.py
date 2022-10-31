import pandas as pd
import numpy as np

def read_data():
    df = pd.read_csv("./data/Products.csv", lineterminator="\n")
    clean_prices = df["prices"].str.replace("£", "")
    print(clean_prices)

read_data()