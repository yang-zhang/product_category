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
        "--model_name_or_path", type=str, default="distilbert-base-cased"
    )
    parser.add_argument(
        "--trained_model_path",
        type=str,
        help=("Model used to predict."),
        default="../data/transformer_20210307D3.ckpt",
    )
    parser.add_argument("--data_file_path", type=str)
    parser.add_argument("--freeze_bert", action="store_true")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--min_products_for_category", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=256)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", action="store_true")

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
    pretrained_model = PCModel.load_from_checkpoint(args.trained_model_path)
    pcmodel = PCModel(
        model_name_or_path=args.model_name_or_path,
        freeze_bert=args.freeze_bert,
        num_classes=data_module.num_classes,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
    )
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
