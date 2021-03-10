from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pc_model import PCDataModule, PCModel
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def cli_main():

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="distilbert-base-cased",
    )
    parser.add_argument(
        "--transfer_learn",
        help="Wether to use transfer learning based on a pretrained model.",
        action="store_true",
    )
    parser.add_argument(
        "--trained_model_path",
        type=str,
        help=("Model used to predict."),
        default="../data/transformer_20210307D3.ckpt",
    )

    parser.add_argument(
        "--data_file_path",
        help="Path to training data file. "
        "Data file should be csv with 3 columns: category (categories separated by '|'),title (str),is_validation (0 or 1). "
        "e.g.: Sports & Outdoors|Outdoor Recreation|Cycling|Clothing|Men|Shorts,Louis Garneau Men's Neo Power Motion Bike Shorts,1",
        type=str,
    )
    parser.add_argument(
        "--freeze_bert",
        help="Whether to freeze the pretrained model.",
        action="store_true",
    )
    parser.add_argument(
        "--max_seq_length",
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--min_products_for_category",
        help="Minimum number of products for a category to be considered in the model.",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--train_batch_size",
        help="How many samples per batch to load for train dataloader.",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--val_batch_size",
        help="How many samples per batch to load for validation dataloader.",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--dataloader_num_workers",
        help="How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--pin_memory",
        help="Wether to use pin_memory in pytorch dataloader. If True, the data loader will copy Tensors into CUDA pinned memory before returning them.",
        action="store_true",
    )

    parser = pl.Trainer.add_argparse_args(parser)
    parser = PCModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    data_module = PCDataModule(
        model_name_or_path=args.model_name_or_path,
        data_file_path=args.data_file_path,
        min_products_for_category=args.min_products_for_category,
        max_seq_length=args.max_seq_length,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
    )
    data_module.prepare_data()
    data_module.setup()

    # ------------
    # model
    # ------------
    pcmodel = PCModel(
        model_name_or_path=args.model_name_or_path,
        freeze_bert=args.freeze_bert,
        num_classes=data_module.num_classes,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
    )
    if args.transfer_learn:
        pretrained_model = PCModel.load_from_checkpoint(args.trained_model_path)
        pcmodel.bert = pretrained_model.bert
    pcmodel.prepare_data()
    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[EarlyStopping(monitor="valid_loss")],
        stochastic_weight_avg=True,
    )
    trainer.fit(pcmodel, data_module)


if __name__ == "__main__":
    cli_main()
