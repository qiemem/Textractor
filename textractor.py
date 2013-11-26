import fileinput
from collections import defaultdict

def make_tuples(tuple_size, sentences):
    """
    Iterates through the words in the sentences, yielding them as tuples of
    size tuple_size.
    
    >>> list(make_tuples(3, ['hurray for information extraction', 'text mining is super fun']))
    [('hurray', 'for', 'information'),
     ('for', 'information', 'extraction'),
     ('text', 'mining', 'is'),
     ('mining', 'is', 'super'),
     ('is', 'super', 'fun')]
    """
    for sentence in sentences:
        words = sentence.split()
        for i, w in enumerate(words[:-(tuple_size - 1)]):
            yield tuple(words[i:i+tuple_size])


if __name__ == '__main__':
    vecs = defaultdict(lambda : [0]*100)
    for w1, w2 in make_tuples(2, fileinput.input()):
        vecs[w1][hash(w2) % 100] += 1
        vecs[w2][hash(w1) % 100] += 1
    for w, v in vecs.iteritems():
        print(w + ' ' + ' '.join(map(str, v)))

