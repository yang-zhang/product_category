# Product Category
Product category prediction model trained based on [Amazon product data](http://jmcauley.ucsd.edu/data/amazon/).

## Installation
```Bash
pip install TODO
```
## Inference
###
```Bash
python product_category/predict_single.py -h
TODO
```

## Training
### Setup Development Environment
```Bash
make create_environment
conda activate product_category
make requirements
```

Organize data `data/example_data.csv`
```
category,title,is_validation
Sports & Outdoors|Outdoor Recreation|Cycling|Clothing|Men|Shorts,Louis Garneau Men's Neo Power Motion Bike Shorts,1
"Clothing, Shoes & Jewelry|Novelty & More|Clothing|Novelty",Nirvana Men's Short Sleeve Many Smiles T-Shirt Shirt,0
Grocery & Gourmet Food|Snack Foods|Chips & Crisps|Tortilla,Doritos Tapatio Salsa Picante Hot Sauce Flavor Chips 7 5/8 oz Bag (Pack of 1),0
"Clothing, Shoes & Jewelry|Women|Shoes|Boots|Synthetic|Synthetic sole|Vegan Friendly",SODA Womens Dome-H Boot,1
"Clothing, Shoes & Jewelry|Baby|Baby Girls|Accessories|Socks",Lian LifeStyle Unisex Children 2 Pairs Knee High Wool Boot Socks MFS02 3 Sizes 14 Colors,0
Sports & Outdoors|Outdoor Recreation|Camping & Hiking,Folding Pot Stabilizer,0
```