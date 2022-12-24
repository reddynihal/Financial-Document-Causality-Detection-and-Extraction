import argparse
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--gold', type=str, help='gold file', required=True)
    parser.add_argument('--pred', type=str, help='predicted file', required=True)
    parser.add_argument('--out', type=str, help='output filename', required=True)

    args = parser.parse_args()

    gold = pd.read_csv(args.gold, sep="; ", dtype={"Index": object})
    pred = pd.read_csv(args.pred)

    if "Gold" in gold.columns:
        del gold["Gold"]

    gold["Prediction"] = pred.pred.apply(lambda x: " " + str(x))
    gold["Text"] = gold.Text.apply(lambda x: " " + str(x))

    gold.to_csv(args.out, sep=';', index=False)
