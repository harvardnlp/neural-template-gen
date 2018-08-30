from collections import defaultdict

# leaves out familyFriendly, which is a binary thing...
e2e_keys = ["name", "eatType", "food", "priceRange", "customerrating", "area", "near"]
e2e_key2idx = dict((key, i) for i, key in enumerate(e2e_keys))

def get_e2e_fields(tokes, keys=None):
    """
    assumes a key only appears once per line...
    returns keyname -> list of words dict
    """
    if keys is None:
        keys = e2e_keys
    fields = defaultdict(list)
    state = None
    for toke in tokes:
        if "__start" in toke:
            for key in keys:
                if toke == "__start_%s__" % key:
                    assert state is None
                    state = key
        elif "__end" in toke:
            for key in keys:
                if toke == "__end_%s__" % key:
                    assert state == key
                    state = None
        elif state is not None:
            fields[state].append(toke)

    return fields

def get_e2e_poswrds(tokes):
    """
    assumes a key only appears once per line...
    returns (key, num) -> word
    """
    fields = {}
    state, num = None, 1 # 1-idx the numbering
    for toke in tokes:
        if "__start" in toke:
            assert state is None
            state = toke[7:-2]
        elif "__end" in toke:
            state, num = None, 1
        elif state is not None:
            fields[state, num] = toke
            num += 1
    return fields


def get_wikibio_fields(tokes, keep_splits=None):
    """
    key -> list of words
    """
    fields = defaultdict(list)
    for toke in tokes:
        try:
            fullkey, val = toke.split(':')
        except ValueError:
            ugh = toke.split(':') # must be colons in the val
            fullkey = ugh[0]
            val = ''.join(ugh[1:])
        if val == "<none>":
            continue
        #try:
        keypieces = fullkey.split('_')
        if len(keypieces) == 1:
            key = fullkey
        else:
            keynum = keypieces[-1]
            key = '_'.join(keypieces[:-1])
            #key, keynum = fullkey.split('_')
        #except ValueError:
        #    key = fullkey
        if keep_splits is None or key not in keep_splits:
            fields[key].append(val) # assuming keys are ordered...
        else:
            fields[fullkey].append(val)
    return fields


def get_wikibio_poswrds(tokes):
    """
    (key, num) -> word
    """
    fields = {}
    for toke in tokes:
        try:
            fullkey, val = toke.split(':')
        except ValueError:
            ugh = toke.split(':') # must be colons in the val
            fullkey = ugh[0]
            val = ''.join(ugh[1:])
        if val == "<none>":
            continue
        #try:
        keypieces = fullkey.split('_')
        if len(keypieces) == 1:
            key = fullkey
            #keynum = '0'
            keynum = 1
        else:
            keynum = int(keypieces[-1])
            key = '_'.join(keypieces[:-1])
        fields[key, keynum] = val
    return fields
