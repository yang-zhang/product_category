import pandas as pd
import gzip
import json
import numpy as np
import re

# request and download data from https://nijianmo.github.io/amazon/index.html
pdata = "../data/amazon_meta_data"  # folder for downloaded data
fnms = [
    "meta_AMAZON_FASHION.json.gz",
    "meta_All_Beauty.json.gz",
    "meta_Appliances.json.gz",
    "meta_Arts_Crafts_and_Sewing.json.gz",
    "meta_Automotive.json.gz",
    "meta_Books.json.gz",
    "meta_CDs_and_Vinyl.json.gz",
    "meta_Cell_Phones_and_Accessories.json.gz",
    "meta_Clothing_Shoes_and_Jewelry.json.gz",
    "meta_Digital_Music.json.gz",
    "meta_Electronics.json.gz",
    "meta_Gift_Cards.json.gz",
    "meta_Grocery_and_Gourmet_Food.json.gz",
    "meta_Home_and_Kitchen.json.gz",
    "meta_Industrial_and_Scientific.json.gz",
    "meta_Kindle_Store.json.gz",
    "meta_Luxury_Beauty.json.gz",
    "meta_Magazine_Subscriptions.json.gz",
    "meta_Movies_and_TV.json.gz",
    "meta_Musical_Instruments.json.gz",
    "meta_Office_Products.json.gz",
    "meta_Patio_Lawn_and_Garden.json.gz",
    "meta_Pet_Supplies.json.gz",
    "meta_Prime_Pantry.json.gz",
    "meta_Software.json.gz",
    "meta_Sports_and_Outdoors.json.gz",
    "meta_Tools_and_Home_Improvement.json.gz",
    "meta_Toys_and_Games.json.gz",
    "meta_Video_Games.json.gz",
]
domains = [o.strip("meta_").strip(".json.gz") for o in fnms]
dmn2fnm = dict(zip(domains, fnms))

KEYS2USE = set(["category", "description", "title", "brand", "feature", "asin"])
# load the meta data
def get_meta_data(domain, nrows=None):
    keys2use = KEYS2USE
    fnm = dmn2fnm[domain]
    data = []
    with gzip.open(f"{pdata}/{fnm}") as f:
        for i, l in enumerate(f):
            dat = json.loads(l.strip())
            dat = {k: v for k, v in dat.items() if k in keys2use}
            data.append(dat)
            if nrows and i > nrows:
                break

    # convert list into pandas dataframe
    df = pd.DataFrame.from_dict(data)

    ### remove rows with unformatted title (i.e. some 'title' may still contain html style content)
    df3 = df.fillna("")
    df4 = df3[df3.title.str.contains("getTime")]  # unformatted rows
    df5 = df3[~df3.title.str.contains("getTime")]  # filter those unformatted rows
    return df5


# preprocess data
dfs = []
for dmn in domains:
    df = get_meta_data(dmn, nrows=None)
    if set(df.columns) != KEYS2USE:
        continue
    df = df[df.category.apply(len) > 0]
    if len(df) == 0:
        continue
    df["domain"] = dmn
    dfs.append(df)
df = pd.concat(dfs)

# process category
def try2eval(x):
    try:
        x = eval(x)
        x = (o.replace("|", "") for o in x)
        return "|".join(x)
    except SyntaxError:
        return ""


df.category = df.category.astype(str)
df.category = df.category.apply(try2eval)

# clean up text
def cleanhtml(raw_html):
    cleanr = re.compile("<.*?>")
    cleantext = re.sub(cleanr, "", raw_html)
    return cleantext


df.fillna("", inplace=True)
for col in ["description", "feature"]:
    df[col] = df[col].apply(lambda x: "\n".join(x) if len(x) else "")

for col in [
    "description",
    "title",
    "brand",
    "feature",
]:
    df[col] = df[col].apply(cleanhtml)

df["txt"] = df.title + " " + df.brand + " " + df.description + " " + df.feature

# train val split
np.random.seed(101)
df["is_validation"] = np.random.choice(2, size=len(df), p=[0.85, 0.15])

df.to_csv(f"../data/data_processed.csv", index=False)
