import fileinput

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
    for t in make_tuples(3, fileinput.input()):
        print(t)

