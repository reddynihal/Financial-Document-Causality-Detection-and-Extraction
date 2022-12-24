from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix

import argparse
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-tp', '--test_prefix', type=str, default="", dest="test_prefix",
                        help= 'before cross val id')
    parser.add_argument('-ts', '--test_suffix', type=str, default=".csv", dest="test_suffix",
                        help= 'after cross val id')
    parser.add_argument('-pp', '--pred_prefix', type=str, default="", dest="pred_prefix",
                        help= 'before cross val id')
    parser.add_argument('-ps', '--pred_suffix', type=str, default="_4.csv", dest="pred_suffix",
                        help= 'after cross val id')
    parser.add_argument('-f', '--folds', type=int, default=5, dest="folds")

    args = parser.parse_args()

    test_l = [pd.read_csv(args.test_prefix + str(i) + args.test_suffix) for i in range(args.folds)]
    pred_l = [pd.read_csv(args.pred_prefix + str(i) + args.pred_suffix) for i in range(args.folds)]

    test = pd.concat(test_l)
    pred = pd.concat(pred_l)
    test["pred"] = pred.pred

    print(classification_report(test.label, test.pred))
    print()
    print(confusion_matrix(test.label, test.pred, labels=list(range(7))))
    print()

    mlb = MultiLabelBinarizer()
    true_site_level = test.groupby(["id"]).label.unique().apply(lambda x: [i for i in x if i > 0])
    pred_site_level = test.groupby(["id"]).pred.unique().apply(lambda x: [i for i in x if i > 0])
    test.groupby(["id"]).pred.unique().apply(lambda x: [i for i in x if i > 0])
    true_ohe = mlb.fit_transform(true_site_level)
    pred_ohe = mlb.transform(pred_site_level)
    print(classification_report(true_ohe, pred_ohe, target_names=[str(i+1) for i in range(6)]))

