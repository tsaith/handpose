import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def natural_sort(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def natural_sorted(l):
    """ Sort the given list in the way that humans expect.
    """
    return sorted(l, key=alphanum_key)
