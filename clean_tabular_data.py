import pandas as pd

def read_data():
    df = pd.read_csv("./data/Products.csv", lineterminator="\n")
    clean_prices = df["price"].str.replace("£", "")

read_data()