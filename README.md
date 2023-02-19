# Facebook Marketplace Recommendation Ranking System

## **Milestone 1**
---
The data was cleaned using the `pandas` and `PILLOW` libraries. The csv file was read into a variable called `df` using the `pd.read_csv()` function. The prices column contains non-numerical values which need to be cleaned and to do this the `.replace()` function was applied to the column replacing the £ symbol with nothing.

The image dataset also needed to be cleaned to be used for analysis using the `PILLOW` library. The images are set a size of 512px x 512px and rescaled to the new size.

---
## **Milestone 2**
---


---
## **Milestone 3**
---


---
## **Milestone 4**
---


---

## **Notes**
---
Only 12604 images have been labelled out of the 12668 raw images
There are 64 images that are not listed in Images.csv
There are 7051 unique items in the Images.csv file but 7155 unique items in the Products.csv file