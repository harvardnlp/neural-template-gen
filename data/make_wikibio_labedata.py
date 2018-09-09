import os
import sys
import torch

from utils import get_wikibio_fields

train_dir = "wikipedia-biography-dataset/train"
val_dir = "wikipedia-biography-dataset/valid"

punctuation = set(['.', '!', ',', ';', ':', '?', '--', '-rrb-', '-lrb-'])

# from wikipedia
prepositions = set(['aboard', 'about', 'above', 'absent', 'across', 'after', 'against', 'along', 'alongside', 'amid', 'among',
                    'apropos', 'apud', 'around', 'as', 'astride', 'at', 'atop', 'bar', 'before', 'behind', 'below', 'beneath',
                    'beside', 'besides', 'between', 'beyond', 'but', 'by', 'chez', 'circa', 'come', 'despite', 'down', 'during',
                    'except', 'for', 'from', 'in', 'inside', 'into', 'less', 'like', 'minus', 'near', 'notwithstanding', 'of',
                    'off', 'on', 'onto', 'opposite', 'out', 'outside', 'over', 'pace', 'past', 'per', 'plus', 'post', 'pre',
                    'pro', 'qua', 're', 'sans', 'save', 'short', 'since', 'than', 'through', 'throughout', 'till', 'to', 'toward',
                    'under', 'underneath', 'unlike', 'until', 'unto', 'up', 'upon', 'upside', 'versus', 'via', 'vice', 'aboard',
                    'about', 'above', 'absent', 'across', 'after', 'against', 'along', 'alongside', 'amid', 'among', 'apropos',
                    'apud', 'around', 'as', 'astride', 'at', 'atop', 'bar', 'before', 'behind', 'below', 'beneath', 'beside', 'besides',
                    'between', 'beyond', 'but', 'by', 'chez', 'circa', 'come', 'despite', 'down', 'during', 'except', 'for', 'from', 'in',
                    'inside', 'into', 'less', 'like', 'minus', 'near', 'notwithstanding', 'of', 'off', 'on', 'onto', 'opposite', 'out',
                    'outside', 'over', 'pace', 'past', 'per', 'plus', 'post', 'pre', 'pro', 'qua', 're', 'sans', 'save', 'short', 'since',
                    'than', 'through', 'throughout', 'till', 'to', 'toward', 'under', 'underneath', 'unlike', 'until', 'unto', 'up', 'upon',
                    'upside', 'versus', 'via', 'vice', 'with', 'within', 'without', 'worth'])


splitters = set(['and', ',', 'or', 'of', 'for', '--', 'also'])

goodsplitters = set([',', 'of', 'for', '--', 'also']) # leaves out and and or

def splitphrs(tokes, l, r, max_phrs_len, labelist):
    if r-l <= max_phrs_len:
        labelist.append((l, r, 0))
    else:
        i = r-1
        found_a_split = False
        while i > l:
            if tokes[i] in goodsplitters or tokes[i] in prepositions:
                splitphrs(tokes, l, i, max_phrs_len, labelist)
                if i < r-1:
                    splitphrs(tokes, i+1, r, max_phrs_len, labelist)
                found_a_split = True
                break
            i -= 1
        if not found_a_split: # add back in and and or
            i = r-1
            while i > l:
                if tokes[i] in splitters or tokes[i] in prepositions:
                    splitphrs(tokes, l, i, max_phrs_len, labelist)
                    if i < r-1:
                        splitphrs(tokes, i+1, r, max_phrs_len, labelist)
                    found_a_split = True
                    break
                i -= 1
        if not found_a_split: # just do something
            i = r-1
            while i >= l:
                max_len = min(max_phrs_len, i-l+1)
                labelist.append((i-max_len+1, i+1, 0))
                i = i-max_len


def stupid_search(tokes, fields):
    """
    greedily assigns longest labels to spans from right to left
    """
    PFL = 4
    labels = []
    i = len(tokes)
    wordsets = [set(toke for toke in v if toke not in punctuation) for k, v in fields.iteritems()]
    pfxsets = [set(toke[:PFL] for toke in v if toke not in punctuation) for k, v in fields.iteritems()]
    while i > 0:
        matched = False
        if tokes[i-1] in punctuation:
            labels.append((i-1, i, 0)) # all punctuation
            i -= 1
            continue
        if tokes[i-1] in punctuation or tokes[i-1] in prepositions or tokes[i-1] in splitters:
            i -= 1
            continue
        for j in xrange(i):
            if tokes[j] in punctuation or tokes[j] in prepositions or tokes[j] in splitters:
                continue
            # then check if it matches stuff in the table
            tokeset = set(toke for toke in tokes[j:i] if toke not in punctuation)
            for vset in wordsets:
                if tokeset == vset or (tokeset.issubset(vset) and len(tokeset) > 1):
                    if i - j > max_phrs_len:
                        nugz = []
                        splitphrs(tokes, j, i, max_phrs_len, nugz)
                        labels.extend(nugz)
                    else:
                        labels.append((j, i, 0))
                    i = j
                    matched = True
                    break
            if matched:
                break
            pset = set(toke[:PFL] for toke in tokes[j:i] if toke not in punctuation)
            for pfxset in pfxsets:
                if pset == pfxset or (pset.issubset(pfxset)and len(pset) > 1):
                    if i - j > max_phrs_len:
                        nugz = []
                        splitphrs(tokes, j, i, max_phrs_len, nugz)
                        labels.extend(nugz)
                    else:
                        labels.append((j, i, 0))
                    i = j
                    matched = True
                    break
            if matched:
                break
        if not matched:
            i -= 1
    labels.sort(key=lambda x: x[0])
    return labels

def print_data(direc):
    fis = os.listdir(direc)
    srcfi = [fi for fi in fis if fi.endswith('.box')][0]
    tgtfi = [fi for fi in fis if fi.endswith('.sent')][0]
    nbfi = [fi for fi in fis if fi.endswith('.nb')][0]

    with open(os.path.join(direc, srcfi)) as f:
        srclines = f.readlines()
    with open(os.path.join(direc, nbfi)) as f:
        nbs = [0]
        [nbs.append(int(line.strip())) for line in f.readlines()]
        nbs = set(torch.Tensor(nbs).cumsum(0))

    tgtlines = []
    with open(os.path.join(direc, tgtfi)) as f:
        for i, tgtline in enumerate(f):
            if i in nbs:
                tgtlines.append(tgtline)

    assert len(srclines) == len(tgtlines)
    for i in xrange(len(srclines)):
        fields = get_wikibio_fields(srclines[i].strip().split())
        tgttokes = tgtlines[i].strip().split()
        labels = stupid_search(tgttokes, fields)
        labels = [(str(tup[0]), str(tup[1]), str(tup[2])) for tup in labels]
        # add eos stuff
        tgttokes.append("<eos>")
        labels.append((str(len(tgttokes)-1), str(len(tgttokes)), '0')) # label doesn't matter

        labelstr = " ".join([','.join(label) for label in labels])
        sentstr = " ".join(tgttokes)

        outline = "%s|||%s" % (sentstr, labelstr)
        print outline

if __name__ == "__main__":
    max_phrs_len = int(sys.argv[2])
    if sys.argv[1] == "train":
        print_data(train_dir)
    elif sys.argv[1] == "valid":
        print_data(val_dir)
    else:
        assert False
