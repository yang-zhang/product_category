# Deep Learning Model for Product Category Prediction
Product category prediction model built with:
- [pytorch](https://github.com/pytorch/pytorch)
- [huggingface/transformers](https://github.com/huggingface/transformers)
- [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) 

and trained using [Amazon product data](http://jmcauley.ucsd.edu/data/amazon/). 

This library supports
- Predicting categories using the pretrained model.
- Training from scratch, with a transformers model as the starting point.
- Transfer learning from the pretrained model.

## Pretrained model
The pretrained model is trained using product category and title in the metadata [Amazon product data](http://jmcauley.ucsd.edu/data/amazon/). Each product can have multiple categories.
We sample 500K products (85% for train; 15% for validation) to train the model, which resulted in ~1900 categories.
We use [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) to train a multilabel classification model with the pretrained `distilbert-base-cased` model from [huggingface/transformers](https://github.com/huggingface/transformers) as the starting point.
This library supports 
1. directly using this pretrained model to predict the ~1900 categories from an input product title or description; 
2. using this pretrained model as a starting point to do transfer learning and train a category prediction model on your own categories, as long as you provide training data in the format described below.

You can also train a model from scratch without using this pretrained model, but instead with a transformers model as the starting point.

### Download Pretrained Model
Download the pretrained model to `data` folder:
```Bash
wget https://github.com/yang-zhang/product_category/releases/download/v0.0.1/transformer_20210307D3.ckpt -P data
```

## Installation
```Bash
pip install product-category
```

## Predict with Pre-trained Model
```
python product_category/predict.py -h
usage: predict.py [-h] -t TEXT [--trained_model_path TRAINED_MODEL_PATH]
                  [--i2cat_path I2CAT_PATH] [--tokenizer_name TOKENIZER_NAME]
                  [--topn TOPN]

optional arguments:
  -h, --help            show this help message and exit
  -t TEXT, --text TEXT  Product info text to predict.
  --trained_model_path TRAINED_MODEL_PATH
                        Model used to predict.
  --i2cat_path I2CAT_PATH
                        File name for the ordered list of categories. Each
                        line for one category.
  --tokenizer_name TOKENIZER_NAME
                        Tokenizer name.
  --topn TOPN           Number of top predicted categories to display.
  ```

For example:
```
python predict.py -t "Lykmera Famous TikTok Leggings, High Waist Yoga Pants for Women, Booty Bubble Butt Lifting Workout Running Tights"

Sports & Outdoors: 0.997
Sports & Fitness: 0.994
Exercise & Fitness: 0.980
Clothing: 0.961
Yoga: 0.905
```

## Training Data Format
Training data file should be csv with 3 columns: `category` (categories separated by '|'), `title` (str), `is_validation` (0 or 1). Similar to `data/example_data.csv`. 
```
category,title,is_validation
Sports & Outdoors|Outdoor Recreation|Cycling|Clothing|Men|Shorts,Louis Garneau Men's Neo Power Motion Bike Shorts,1
"Clothing, Shoes & Jewelry|Novelty & More|Clothing|Novelty",Nirvana Men's Short Sleeve Many Smiles T-Shirt Shirt,0
Grocery & Gourmet Food|Snack Foods|Chips & Crisps|Tortilla,Doritos Tapatio Salsa Picante Hot Sauce Flavor Chips 7 5/8 oz Bag (Pack of 1),0
"Clothing, Shoes & Jewelry|Women|Shoes|Boots|Synthetic|Synthetic sole|Vegan Friendly",SODA Womens Dome-H Boot,1
Sports & Outdoors|Outdoor Recreation|Camping & Hiking,Folding Pot Stabilizer,0
```

## Training
Below are a subset of options for `training.py`. 
Run `python train.py -h` to see full help list, which includes more options for [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) functionalities.
```
python train.py -h
usage: train.py [-h] [--model_name_or_path MODEL_NAME_OR_PATH]
                [--transfer_learn] [--trained_model_path TRAINED_MODEL_PATH]
                [--data_file_path DATA_FILE_PATH] [--freeze_bert]
                [--max_seq_length MAX_SEQ_LENGTH]
                [--min_products_for_category MIN_PRODUCTS_FOR_CATEGORY]
                [--train_batch_size TRAIN_BATCH_SIZE]
                [--val_batch_size VAL_BATCH_SIZE]
                [--dataloader_num_workers DATALOADER_NUM_WORKERS]
                [--pin_memory] [--logger [LOGGER]]
                [--learning_rate LEARNING_RATE] 

optional arguments:
  -h, --help            show this help message and exit
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to pretrained model or model identifier from
                        huggingface.co/models.
  --transfer_learn      Wether to use transfer learning based on a pretrained
                        model.
  --trained_model_path TRAINED_MODEL_PATH
                        Model used to predict.
  --data_file_path DATA_FILE_PATH
                        Path to training data file. Data file should be csv
                        with 3 columns: category (categories separated by
                        '|'),title (str),is_validation (0 or 1). e.g.: Sports
                        & Outdoors|Outdoor
                        Recreation|Cycling|Clothing|Men|Shorts,Louis Garneau
                        Men's Neo Power Motion Bike Shorts,1
  --freeze_bert         Whether to freeze the pretrained model.
  --max_seq_length MAX_SEQ_LENGTH
                        The maximum total input sequence length after
                        tokenization. Sequences longer than this will be
                        truncated, sequences shorter will be padded.
  --min_products_for_category MIN_PRODUCTS_FOR_CATEGORY
                        Minimum number of products for a category to be
                        considered in the model.
  --train_batch_size TRAIN_BATCH_SIZE
                        How many samples per batch to load for train
                        dataloader.
  --val_batch_size VAL_BATCH_SIZE
                        How many samples per batch to load for validation
                        dataloader.
  --dataloader_num_workers DATALOADER_NUM_WORKERS
                        How many subprocesses to use for data loading. 0 means
                        that the data will be loaded in the main process.
  --pin_memory          Wether to use pin_memory in pytorch dataloader. If
                        True, the data loader will copy Tensors into CUDA
                        pinned memory before returning them.
  --learning_rate LEARNING_RATE
                        Learning Rate
```
### Training from Scratch
Training from scratch, with a transformers model as the starting point.

For example:
```
python train.py --data_file_path ../data/sample_data.csv

```
### Transfer Learning from Pre-trained Model
Transfer learning from the pretrained model.

For example:
```
python train.py --transfer_learn --data_file_path ../data/sample_data.csv
```

### Useful Pytorch-Lightning Options
To run with GPU:
```
python train.py --transfer_learn --data_file_path ../data/sample_data.csv --gpus=1
```

To train only a classification head with the transformer backbone frozen:
```
python train.py --transfer_learn --data_file_path ../data/sample_data.csv --freeze_bert
```

To run with GPU, `pin_memory` for dataloader, and limiting maximum training epochs:
```
python train.py --transfer_learn --data_file_path ../data/sample_data.csv --gpus=1 --pin_memory --max_epochs=100
```


## Note 
The pretrained model is trained using [Amazon product data](http://jmcauley.ucsd.edu/data/amazon/), which is for research purpose. Therefore, the pretrained model should  also be used for research purposes.


