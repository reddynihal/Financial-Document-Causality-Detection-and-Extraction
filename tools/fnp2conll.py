from fnp.baseline.task2.utils import *
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split


def write_file(file_name, data):
    with open(file_name, "w") as f:
        for sent in data:
            i = 1

            for token, tag in sent:
                f.write(str(i) + "\t" + token + "\t_\t" + tag + "\t_\t_\t_\t_\t_\t_\n")

                i += 1

            f.write("\n")

def merge_series(hometags):
    for doc in hometags:
        for i in range(1, len(doc)-1):
            if doc[i][1] == "_" and doc[i-1][1] == "C" and doc[i+1][1] == "C":
                doc[i] = (doc[i][0], "C")

            if doc[i][1] == "_" and doc[i-1][1] == "E" and doc[i+1][1] == "E":
                doc[i] = (doc[i][0], "E")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--inrepo', type=str, default="./fnp2020-fincausal-task2.csv", help= 'input repo')
    parser.add_argument('--train_out', type=str, default="./fnp2020-fincausal-task2.train.conllu",
                        help='train output filename')
    parser.add_argument('--test_out', type=str, default="./fnp2020-fincausal-task2.test.conllu",
                        help='test output filename')
    parser.add_argument('--mergeseries', dest="merge_series", action='store_true',
                        help='merge empty tokens between two tagged token')
    parser.set_defaults(merge_series=False)

    args = parser.parse_args()

    df = pd.read_csv(args.inrepo, delimiter=';', header=0)

    lodict_ = []
    for rows in df.itertuples():
        list_ = [rows[2], rows[3], rows[4]]
        map1 = ['sentence', 'cause', 'effect']
        dict_ = s2dict(list_, map1)
        lodict_.append(dict_)

    print('transformation example: ', lodict_[1])

    map_ = [('cause', 'C'), ('effect', 'E')]
    hometags = make_causal_input(lodict_, map_)

    if args.merge_series:
        merge_series(hometags)

    size = 0.2
    seed = 42

    train, test = train_test_split(hometags, test_size=size, random_state=seed)

    write_file(args.train_out, train)
    write_file(args.test_out, test)

