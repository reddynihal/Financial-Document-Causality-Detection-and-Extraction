import argparse
import pyconll

document_start = """
<html>
<head>
<style>
.C {color:red;}
.E {color:blue;}
.sent {border: 1px solid; margin-bottom:10px}
</style>
</head>
<body>"""

document_end = """
</body>
</html>"""


def read_conll(file_name):
    data = pyconll.load_from_file(file_name)

    tags = [
        [
            "<span class='" + (token.upos if token.upos in ["C", "E"] else "O") + "'>" + token.form + "</span>"
            for token in sent
        ]
        for sent in data
    ]

    return tags


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-g', type=str, dest="gold",  required=True, help= 'gold filename')
    parser.add_argument('-p', type=str, dest="pred", required=True, help= 'predicted filename')
    parser.add_argument('-o', type=str, dest="out", required=True, help= 'output html')

    args = parser.parse_args()

    gold = read_conll(args.gold)
    pred = read_conll(args.pred)

    document = document_start

    for g_s, p_s in zip(gold, pred):
        document += "<div class='sent'>"
        document += "<div>" + " ".join(g_s) + "</div>"
        document += "<div>" + " ".join(p_s) + "</div>"
        document += "</div>"

    document += document_end

    with open(args.out, "w") as f:
        f.write(document)
