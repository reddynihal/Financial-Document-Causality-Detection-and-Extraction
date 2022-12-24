import argparse
import pandas as pd
import pyconll

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--gold_csv', type=str, help='gold file', required=True)
    parser.add_argument('--pred_conll', type=str, help='predicted file', required=True)
    parser.add_argument('--pred_csv', type=str, help='output filename', required=True)

    args = parser.parse_args()

    df = pd.read_csv(args.gold_csv, delimiter=';', header=0)
    #df = pd.read_csv(args.gold_csv, delimiter='; ', header=0)
    df_gold = df[["Index", " Text"]]
    pred = pyconll.load_from_file(args.pred_conll)

    df_gold[" Cause"] = [" ".join([token.form for token in sent if token.upos == "C"]) for sent in pred]
    df_gold[" Effect"] = [" ".join([token.form for token in sent if token.upos == "E"]) for sent in pred]

    df_gold[" Cause"] = df_gold[" Cause"].apply(lambda x: " " + str(x))
    df_gold[" Effect"] = df_gold[" Effect"].apply(lambda x: " " + str(x))

    df_gold[" Offset_Sentence2"] = " "
    df_gold[" Offset_Sentence3"] = " "

    df_gold.to_csv(args.pred_csv, sep=';', index=False)
