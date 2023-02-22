import os
import pandas as pd
import csv

def create_labels_for_categories():
    # Creates a file called 'Labels.csv' if it doesn't already exist
    if os.path.isfile("./data/Labels.csv"):
        pass
    else:
        with open('./data/Labels.csv', 'w') as f:
            pass
    
    # Reads in the data as a pandas dataframe
    products_csv_df = pd.read_csv('./data/Products.csv', lineterminator="\n")
    # Grabs all the items in the 'category' column
    categories = products_csv_df["category"]
    categories = categories.tolist()
    all_categories = []
    
    # Loops through all the categories and grabs the first category split by '/'
    for category in categories:
        main_category = category.split("/")[0].rstrip()
        # Appends the category to the empty list if it doesn't already exist
        if main_category in all_categories:
            continue
        else:
            all_categories.append(main_category)
            
    # Creates an encoder by converting the list to a DataFrame and then to a csv
    all_cat_df = pd.DataFrame(all_categories, columns=['Category'])
    all_cat_df.to_csv("./data/Labels.csv", quoting=csv.QUOTE_NONNUMERIC)

def label_each_product():
    # Creates a file called 'LabelledImages.csv' if it doesn't already exist
    if os.path.isfile("./data/LabelledImages.csv"):
        pass
    else:
        with open('./data/LabelledImages.csv', 'w') as f:
            pass
    
    images_csv_df = pd.read_csv('./data/Images.csv', delimiter=',')[['id', 'product_id']]
    products_csv_df = pd.read_csv('./data/Products.csv', lineterminator="\n")[['id', 'category']]
    labels_csv = pd.read_csv('./data/Labels.csv').rename(columns={'Unnamed: 0' : 'idx'}).to_dict()
    encoder_dict = dict((v,k) for k,v in labels_csv['Category'].items())

    labelled_images_df = pd.merge(images_csv_df, products_csv_df, left_on='product_id', right_on='id').drop(['id_y'], axis=1).rename(columns={'id_x': 'id'})
    for category in labelled_images_df['category']:
        main_category = category.split("/")[0].rstrip()
        labelled_images_df = labelled_images_df.replace(category, main_category)

    labelled_images_df = labelled_images_df.replace({'category': encoder_dict})
    labelled_images_df.to_csv('./data/LabelledImages.csv')
    
create_labels_for_categories()
label_each_product()