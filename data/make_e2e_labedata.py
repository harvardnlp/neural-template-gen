import os
import sys

from utils import get_e2e_fields, e2e_key2idx

e2e_train_src = "trainset-source.tok"
e2e_train_tgt = "trainset-target.tok"
e2e_val_src = "devset-source.tok"
e2e_val_tgt = "devset-target.tok"

punctuation = set(['.', '!', ',', ';', ':', '?'])

def get_first_sent_tokes(tokes):
    try:
        first_per = tokes.index('.')
        return tokes[:first_per+1]
    except ValueError:
        return tokes

def stupid_search(tokes, fields):
    """
    greedily assigns longest labels to spans from left to right
    """
    labels = []
    i = 0
    while i < len(tokes):
        matched = False
        for j in xrange(len(tokes), i, -1):
            # first check if it's punctuation
            if all(toke in punctuation for toke in tokes[i:j]):
                labels.append((i, j, len(e2e_key2idx))) # first label after rul labels
                i = j
                matched = True
                break
            # then check if it matches stuff in the table
            for k, v in fields.iteritems():
                # take an uncased match
                if " ".join(tokes[i:j]).lower() == " ".join(v).lower():
                    labels.append((i, j, e2e_key2idx[k]))
                    i = j
                    matched = True
                    break
            if matched:
                break
        if not matched:
            i += 1
    return labels

def print_data(srcfi, tgtfi):
    with open(srcfi) as f1:
        with open(tgtfi) as f2:
            for srcline in f1:
                tgttokes = f2.readline().strip().split()
                senttokes = tgttokes

                fields = get_e2e_fields(srcline.strip().split()) # fieldname -> tokens
                labels = stupid_search(senttokes, fields)
                labels = [(str(tup[0]), str(tup[1]), str(tup[2])) for tup in labels]

                # add eos stuff
                senttokes.append("<eos>")
                labels.append((str(len(senttokes)-1), str(len(senttokes)), '8')) # label doesn't matter

                labelstr = " ".join([','.join(label) for label in labels])
                sentstr = " ".join(senttokes)

                outline = "%s|||%s" % (sentstr, labelstr)
                print outline


if sys.argv[1] == "train":
    print_data(e2e_train_src, e2e_train_tgt)
elif sys.argv[1] == "valid":
    print_data(e2e_val_src, e2e_val_tgt)
else:
    assert False
