import argparse
import pyconll
import numpy as np

from sklearn.metrics import classification_report, precision_recall_fscore_support

from fnp.baseline.task2.utils import *


def read_conll(file_name):
    data = pyconll.load_from_file(file_name)

    tags = [[token.upos if token.upos in ["C", "E"] else "_" for token in sent] for sent in data]

    return tags


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-g', type=str, dest="gold",  required=True, help= 'gold filename')
    parser.add_argument('-p', type=str, dest="pred", required=True, help= 'predicted filename')

    args = parser.parse_args()

    y_test = read_conll(args.gold)
    y_pred = read_conll(args.pred)

    labels = {"C": 1, "E": 2, "_": 0}

    # Convert the sequences of tags into a 1-dimensional array
    predictions = np.array([labels[tag] for row in y_pred for tag in row])
    truths = np.array([labels[tag] for row in y_test for tag in row])
    print(np.sum(truths == predictions) / len(truths))

    # ------------------------------------------------------------------------------------ #
    #                                    Print metrics                                     #
    # -------------------------------------------------------------------------------------#

    # # Print out the classification report
    print('************************ classification report ***************************', '\t')
    print(classification_report(
        truths, predictions,
        target_names=["_", "C", "E"]))

    # # Print out task2 metrics
    print('************************ tasks metrics ***************************', '\t')

    F1metrics = precision_recall_fscore_support(truths, predictions, average='weighted')
    # print results and make tagged sentences
    ll = []
    # for i in range(len(X_test) - 1):
    #     l = defaultdict(list)
    #     for j, (y, x) in enumerate(zip(y_pred[i], list(zip(*[[v for k, v in x.items()] for x in X_test[i]]))[1])):
    #         l.update({x: y})
    #     ll.append(l)

    # nl = []
    # for line, yt, yp, s in zip(ll, y_test, y_pred, X_test_sent):
    #     d_ = defaultdict(list)
    #     d_["origin"] = s
    #     d_["truth"] = yt
    #     d_["pred"] = yp
    #     d_["diverge"] = 0
    #     for k, v in line.items():
    #         d_[v].append(k)
    #     if d_["truth"] != d_["pred"]:
    #         d_["diverge"] = 1
    #     nl.append(d_)

    nl = []
    for yt, yp in zip(y_test, y_pred):
        d_ = defaultdict(list)
        d_["truth"] = yt
        d_["pred"] = yp
        d_["diverge"] = 0
        if d_["truth"] != d_["pred"]:
            d_["diverge"] = 1
        nl.append(d_)

    print('F1score:', F1metrics[2])
    print('Precision: ', F1metrics[1])
    print('Recall: ', F1metrics[0])
    print('exact match: ', sum([i["diverge"] for i in nl if i["diverge"] == 0]), 'over', len(nl), ' total sentences)')

    # # Print out other metrics
    print('************************ crf metrics ***************************', '\t')
