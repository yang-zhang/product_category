from argparse import ArgumentParser
from transformers import AutoTokenizer
from pc_model import PCModel, mk_tensors
import torch
import numpy as np


def predict(texts, pcmodel, tokenizer):
    input_ids, attention_mask = mk_tensors(texts, tokenizer, 128)
    logits = pcmodel(input_ids, attention_mask)
    scores = torch.sigmoid(logits)
    return scores.detach().numpy()


def cli_main():
    parser = ArgumentParser(add_help=True)
    parser.add_argument(
        "-f", "--filename", type=str, help="Text file to predict.", required=True
    )
    parser.add_argument(
        "--trained_model_path",
        type=str,
        help=("Model used to predict."),
        default="../data/transformer_20210307D3.ckpt",
    )
    parser.add_argument(
        "--i2cat_path",
        type=str,
        help=("i2cat."),
        default="../data/i2cat_transformer_20210307D3.txt",
    )
    parser.add_argument("--tokenizer_name", type=str, default="distilbert-base-cased")

    args = parser.parse_args()
    with open(args.i2cat_path) as f:
        i2cat = f.read().splitlines()

    with open(args.filename) as f:
        texts = f.read().splitlines()

    pcmodel = PCModel.load_from_checkpoint(args.trained_model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    scores = predict(texts, pcmodel, tokenizer)
    top_scores = sorted(scores, reverse=True)[: args.topn]
    top_icats = np.argsort(-scores)[: args.topn]
    for i, scr in zip(top_icats, top_scores):
        print(f"{i2cat[i]}: {scr:.3f}")


if __name__ == "__main__":
    cli_main()
