"""
this file modified from the word_language_model example
"""
import os
import torch

from collections import Counter, defaultdict

from data.utils import get_wikibio_poswrds, get_e2e_poswrds

import random
random.seed(1111)

#punctuation = set(['.', '!', ',', ';', ':', '?', '--', '-rrb-', '-lrb-'])
punctuation = set() # i don't know why i was so worried about punctuation

class Dictionary(object):
    def __init__(self, unk_word="<unk>"):
        self.unk_word = unk_word
        self.idx2word = [unk_word, "<pad>", "<bos>", "<eos>"] # OpenNMT constants
        self.word2idx = {word: i for i, word in enumerate(self.idx2word)}

    def add_word(self, word, train=False):
        """
        returns idx of word
        """
        if train and word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word] if word in self.word2idx else self.word2idx[self.unk_word]

    def bulk_add(self, words):
        """
        assumes train=True
        """
        self.idx2word.extend(words)
        self.word2idx = {word: i for i, word in enumerate(self.idx2word)}

    def __len__(self):
        return len(self.idx2word)


class SentenceCorpus(object):
    def __init__(self, path, bsz, thresh=0, add_bos=False, add_eos=False,
                 test=False):
        self.dictionary = Dictionary()
        self.bsz = bsz
        self.wiki = "wiki" in path or "wb" in path

        train_src = os.path.join(path, "src_train.txt")

        if thresh > 0:
            self.get_vocabs(os.path.join(path, 'train.txt'), train_src, thresh=thresh)
            self.ngen_types = len(self.genset) + 4 # assuming didn't encounter any special tokens
            add_to_dict = False
        else:
            add_to_dict = True
        trsents, trlabels, trfeats, trlocs, inps = self.tokenize(
            os.path.join(path, 'train.txt'), train_src, add_to_dict=add_to_dict,
            add_bos=add_bos, add_eos=add_eos)
        print "using vocabulary of size:", len(self.dictionary)

        print self.ngen_types, "gen word types"
        self.train, self.train_mb2linenos = self.minibatchify(
            trsents, trlabels, trfeats, trlocs, inps, bsz) # list of minibatches

        if (os.path.isfile(os.path.join(path, 'valid.txt'))
                or os.path.isfile(os.path.join(path, 'test.txt'))):
            if not test:
                val_src = os.path.join(path, "src_valid.txt")
                vsents, vlabels, vfeats, vlocs, vinps = self.tokenize(
                    os.path.join(path, 'valid.txt'), val_src, add_to_dict=False,
                    add_bos=add_bos, add_eos=add_eos)
            else:
                print "using test data and whatnot...."
                test_src = os.path.join(path, "src_test.txt")
                vsents, vlabels, vfeats, vlocs, vinps = self.tokenize(
                    os.path.join(path, 'test.txt'), test_src, add_to_dict=False,
                    add_bos=add_bos, add_eos=add_eos)
            self.valid, self.val_mb2linenos = self.minibatchify(
                vsents, vlabels, vfeats, vlocs, vinps, bsz)


    def get_vocabs(self, path, src_path, thresh=2):
        """unks words occurring <= thresh times"""
        tgt_voc = Counter()
        assert os.path.exists(path)

        linewords = []
        with open(src_path, 'r') as f:
            for line in f:
                tokes = line.strip().split()
                if self.wiki:
                    fields = get_wikibio_poswrds(tokes) #key, pos -> wrd
                else:
                    fields = get_e2e_poswrds(tokes) # key, pos -> wrd
                fieldvals = fields.values()
                tgt_voc.update(fieldvals)
                linewords.append(set(wrd for wrd in fieldvals
                                     if wrd not in punctuation))
                tgt_voc.update([k for k, idx in fields])
                tgt_voc.update([idx for k, idx in fields])

        genwords = Counter()
        # Add words to the dictionary
        with open(path, 'r') as f:
            #tokens = 0
            for l, line in enumerate(f):
                words, spanlabels = line.strip().split('|||')
                words = words.split()
                genwords.update([wrd for wrd in words if wrd not in linewords[l]])
                tgt_voc.update(words)

        # prune
        # N.B. it's possible a word appears enough times in total but not in genwords
        # so we need separate unking for generation
        #print "comeon", "aerobatic" in genwords
        for cntr in [tgt_voc, genwords]:
            for k in cntr.keys():
                if cntr[k] <= thresh:
                    del cntr[k]

        self.genset = set(genwords.keys())
        tgtkeys = tgt_voc.keys()
        # make sure gen stuff is first
        tgtkeys.sort(key=lambda x: -(x in self.genset))
        self.dictionary.bulk_add(tgtkeys)
        # make sure we did everything right (assuming didn't encounter any special tokens)
        assert self.dictionary.idx2word[4 + len(self.genset) - 1] in self.genset
        assert self.dictionary.idx2word[4 + len(self.genset)] not in self.genset
        self.dictionary.add_word("<ncf1>", train=True)
        self.dictionary.add_word("<ncf2>", train=True)
        self.dictionary.add_word("<ncf3>", train=True)
        self.dictionary.add_word("<go>", train=True)
        self.dictionary.add_word("<stop>", train=True)


    def tokenize(self, path, src_path, add_to_dict=False, add_bos=False, add_eos=False):
        """Assumes fmt is sentence|||s1,e1,k1 s2,e2,k2 ...."""
        assert os.path.exists(path)

        src_feats, src_wrd2idxs, src_wrd2fields = [], [], []
        w2i = self.dictionary.word2idx
        with open(src_path, 'r') as f:
            for line in f:
                tokes = line.strip().split()
                #fields = get_e2e_fields(tokes, keys=self.e2e_keys) #keyname -> list of words
                if self.wiki:
                    fields = get_wikibio_poswrds(tokes) #key, pos -> wrd
                else:
                    fields = get_e2e_poswrds(tokes) # key, pos -> wrd
                # wrd2things will be unordered
                feats, wrd2idxs, wrd2fields = [], defaultdict(list), defaultdict(list)
                # get total number of words per field
                fld_cntr = Counter([key for key, _ in fields])
                for (k, idx), wrd in fields.iteritems():
                    if k in w2i:
                        featrow = [self.dictionary.add_word(k, add_to_dict),
                                   self.dictionary.add_word(idx, add_to_dict),
                                   self.dictionary.add_word(wrd, add_to_dict)]
                        wrd2idxs[wrd].append(len(feats))
                        #nflds = self.dictionary.add_word(fld_cntr[k], add_to_dict)
                        cheatfeat = w2i["<stop>"] if fld_cntr[k] == idx else w2i["<go>"]
                        wrd2fields[wrd].append((featrow[2], featrow[0], featrow[1], cheatfeat))
                        feats.append(featrow)
                src_wrd2idxs.append(wrd2idxs)
                src_wrd2fields.append(wrd2fields)
                src_feats.append(feats)

        sents, labels, copylocs, inps = [], [], [], []

        # Add words to the dictionary
        tgtline = 0
        with open(path, 'r') as f:
            #tokens = 0
            for line in f:
                words, spanlabels = line.strip().split('|||')
                words = words.split()
                sent, copied, insent = [], [], []
                if add_bos:
                    sent.append(self.dictionary.add_word('<bos>', True))
                for word in words:
                    # sent is just used for targets; we have separate inputs
                    if word in self.genset:
                        sent.append(w2i[word])
                    else:
                        sent.append(w2i["<unk>"])
                    if word not in punctuation and word in src_wrd2idxs[tgtline]:
                        copied.append(src_wrd2idxs[tgtline][word])
                        winps = [[widx, kidx, idxidx, nidx]
                                 for widx, kidx, idxidx, nidx in src_wrd2fields[tgtline][word]]
                        insent.append(winps)
                    else:
                        #assert sent[-1] < self.ngen_types
                        copied.append([-1])
                         # 1 x wrd, tokennum, totalnum
                        #insent.append([[sent[-1], w2i["<ncf1>"], w2i["<ncf2>"]]])
                        insent.append([[sent[-1], w2i["<ncf1>"], w2i["<ncf2>"], w2i["<ncf3>"]]])
                #sent.extend([self.dictionary.add_word(word, add_to_dict) for word in words])
                if add_eos:
                    sent.append(self.dictionary.add_word('<eos>', True))
                labetups = [tupstr.split(',') for tupstr in spanlabels.split()]
                labelist = [(int(tup[0]), int(tup[1]), int(tup[2])) for tup in labetups]
                sents.append(sent)
                labels.append(labelist)
                copylocs.append(copied)
                inps.append(insent)
                tgtline += 1
        assert len(sents) == len(labels)
        assert len(src_feats) == len(sents)
        assert len(copylocs) == len(sents)
        return sents, labels, src_feats, copylocs, inps

    def featurize_tbl(self, fields):
        """
        fields are key, pos -> wrd maps
        returns: nrows x nfeats tensor
        """
        feats = []
        for (k, idx), wrd in fields.iteritems():
            if k in self.dictionary.word2idx:
                featrow = [self.dictionary.add_word(k, False),
                           self.dictionary.add_word(idx, False),
                           self.dictionary.add_word(wrd, False)]
                feats.append(featrow)
        return torch.LongTensor(feats)

    def padded_loc_mb(self, curr_locs):
        """
        curr_locs is a bsz-len list of tgt-len list of locations
        returns:
          a seqlen x bsz x max_locs tensor
        """
        max_locs = max(len(locs) for blocs in curr_locs for locs in blocs)
        for blocs in curr_locs:
            for locs in blocs:
                if len(locs) < max_locs:
                    locs.extend([-1]*(max_locs - len(locs)))
        return torch.LongTensor(curr_locs).transpose(0, 1).contiguous()

    def padded_feat_mb(self, curr_feats):
        """
        curr_feats is a bsz-len list of nrows-len list of features
        returns:
          a bsz x max_nrows x nfeats tensor
        """
        max_rows = max(len(feats) for feats in curr_feats)
        nfeats = len(curr_feats[0][0])
        for feats in curr_feats:
            if len(feats) < max_rows:
                [feats.append([self.dictionary.word2idx["<pad>"] for _ in xrange(nfeats)])
                 for _ in xrange(max_rows - len(feats))]
        return torch.LongTensor(curr_feats)


    def padded_inp_mb(self, curr_inps):
        """
        curr_inps is a bsz-len list of seqlen-len list of nlocs-len list of features
        returns:
          a bsz x seqlen x max_nlocs x nfeats tensor
        """
        max_rows = max(len(feats) for seq in curr_inps for feats in seq)
        nfeats = len(curr_inps[0][0][0])
        for seq in curr_inps:
            for feats in seq:
                if len(feats) < max_rows:
                    # pick random rows
                    randidxs = [random.randint(0, len(feats)-1)
                                for _ in xrange(max_rows - len(feats))]
                    [feats.append(feats[ridx]) for ridx in randidxs]
        return torch.LongTensor(curr_inps)


    def minibatchify(self, sents, labels, feats, locs, inps, bsz):
        """
        this should result in there never being any padding.
        each minibatch is:
          (seqlen x bsz, bsz-length list of lists of (start, end, label) constraints,
           bsz x nfields x nfeats, seqlen x bsz x max_locs, seqlen x bsz x max_locs x nfeats)
        """
        # sort in ascending order
        sents, sorted_idxs = zip(*sorted(zip(sents, range(len(sents))), key=lambda x: len(x[0])))
        minibatches, mb2linenos = [], []
        curr_batch, curr_labels, curr_feats, curr_locs, curr_linenos = [], [], [], [], []
        curr_inps = []
        curr_len = len(sents[0])
        for i in xrange(len(sents)):
            if len(sents[i]) != curr_len or len(curr_batch) == bsz: # we're done
                minibatches.append((torch.LongTensor(curr_batch).t().contiguous(),
                                    curr_labels, self.padded_feat_mb(curr_feats),
                                    self.padded_loc_mb(curr_locs),
                                    self.padded_inp_mb(curr_inps).transpose(0, 1).contiguous()))
                mb2linenos.append(curr_linenos)
                curr_batch = [sents[i]]
                curr_len = len(sents[i])
                curr_labels = [labels[sorted_idxs[i]]]
                curr_feats = [feats[sorted_idxs[i]]]
                curr_locs = [locs[sorted_idxs[i]]]
                curr_inps = [inps[sorted_idxs[i]]]
                curr_linenos = [sorted_idxs[i]]
            else:
                curr_batch.append(sents[i])
                curr_labels.append(labels[sorted_idxs[i]])
                curr_feats.append(feats[sorted_idxs[i]])
                curr_locs.append(locs[sorted_idxs[i]])
                curr_inps.append(inps[sorted_idxs[i]])
                curr_linenos.append(sorted_idxs[i])
        # catch last
        if len(curr_batch) > 0:
            minibatches.append((torch.LongTensor(curr_batch).t().contiguous(),
                                curr_labels, self.padded_feat_mb(curr_feats),
                                self.padded_loc_mb(curr_locs),
                                self.padded_inp_mb(curr_inps).transpose(0, 1).contiguous()))
            mb2linenos.append(curr_linenos)
        return minibatches, mb2linenos
