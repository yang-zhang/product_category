from argparse import ArgumentParser


def predict_single(i2cat, txt):
    return "1"


def cli_main():
    parser = ArgumentParser(add_help=True)
    parser.add_argument(
        "-t", "--text", type=str, help="Text to predict.", required=True
    )
    parser.add_argument(
        "--model_path",
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
    args = parser.parse_args()
    with open(args.i2cat_path) as f:
        i2cat = f.read().splitlines()

    res = predict_single(i2cat, args.text)
    print(res)


if __name__ == "__main__":
    cli_main()
