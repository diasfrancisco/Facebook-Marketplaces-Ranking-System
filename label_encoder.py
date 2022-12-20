import os
import pandas as pd
import csv

def label_encoder():
    # Creates a file called 'Labels.csv' if it doesn't already exist
    if os.path.isfile("./data/Labels.csv"):
        pass
    else:
        with open('./data/Labels.csv', 'w') as f:
            pass
    
    # Reads in the data as a pandas dataframe
    data = pd.read_csv('./data/Products.csv', lineterminator="\n")
    # Grabs all the items in the 'category' column
    categories = data["category"]
    all_categories = []
    
    # Loops through all the categories and grabs the first category split by '/'
    for category in categories:
        main_category = category.split("/")[0]
        # Appends the category to the empty list if it doesn't already exist
        if main_category in all_categories:
            continue
        else:
            all_categories.append(main_category)
            
    # Opens the Labels.csv file and writes to it the encoder for each category
    with open('./data/Labels.csv', 'w') as f:
        writer = csv.writer(f)
        for category, encoder in enumerate(all_categories):
            label = [category, encoder]
            writer.writerow(label)
        
label_encoder()