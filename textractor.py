import fileinput
import argparse
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

def forward_prob(trans_probs, init_probs, emit_probs, sequence):
    """
    trans_probs = state X state matrix
    init_probs = state vector
    emit_probs = state X words
    """
    n = len(init_probs)
    time_state_prob = [[0]*n for _ in sequence]
    for i, prob in enumerate(init_probs):
        time_state_prob[0][i] = prob * emit_probs[i][sequence[0]]

    for i, observed in enumerate(sequence[1:]):
        t = i + 1
        for state in xrange(n):
            obs_prob = emit_probs[state][sequence[t]]
            trans_prob = sum(trans_probs[prev_state][state] * time_state_prob[t-1][prev_state] for prev_state in xrange(n) )
            time_state_prob[t][state] = obs_prob * trans_prob
    return time_state_prob
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Given a bunch of sentences, outputs feature vectors of the words')
    parser.add_argument('-n', default=100, type=int, 
            help='Length of feature vectors (default is 100)')
    parser.add_argument('-f', default='-', type=str, metavar='filename',
            help='File containing sentences to process (defaults to stdin)')

    args = parser.parse_args()
    vecs = defaultdict(lambda : [0]*args.n)
    for w1, w2 in make_tuples(2, fileinput.input(args.f)):
        vecs[w1][hash(w2) % args.n] += 1
        vecs[w2][hash(w1) % args.n] += 1

    for w, v in vecs.iteritems():
        print(w + ' ' + ' '.join(map(str, v)))

