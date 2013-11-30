import fileinput
import argparse
import random
from collections import defaultdict
import itertools
import numpy as np

def normalized(seq):
    total = sum(seq)
    return (x / total for x in seq)

def distribution(size):
    return normalized([random.random() for i in xrange(size)])

def random_hmm(num_states, observables):
    states = range(num_states)
    trans_probs = [list(distribution(num_states)) for s in states]
    init_probs = list(distribution(num_states))
    emit_probs = [dict(zip(observables, list(distribution(len(observables)))))
                  for s in states]
    return HMM(trans_probs, init_probs, emit_probs)

class HMM(object):
    def __init__(self, trans_probs, init_probs, emit_probs):
        self.trans_probs = trans_probs
        self.init_probs = init_probs
        self.emit_probs = emit_probs




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

def enum_range(seq, start=0, stop=None, step=1):
    """
    Like `enumerate`, but lets you specify start, stop, and step used in
    iteration.
    For instance:
    >>> list(enum_range(['a', 'b', 'c'], -1, 0, -1))
    [(2, 'c'), (1, 'b'), (0, 'c')]
    """

    if start < 0:
        start = len(seq) + start
    if stop == None:
        stop = len(seq)
    if stop < 0:
        stop = len(seq) + stop

    for i in xrange(start, stop, step):
        yield i, seq[i]

def forward_prob(trans_probs, init_probs, emit_probs, sequence):
    """
    For each state at each time step, calculates the probability that the hmm
    would produce the given sequence of observations up to that time step and
    land in that state.
    trans_probs - The transition probabilities between states as 
        a list of lists. trans_prob[source state][dest state]
    init_probs - The probabilities of beginning in any given state as vector
        indexed by state.
    emit_probs - The probabilities that each state will produce each 
        observation as a list of lists (or list of dicts).
        emit_probs[state][observation]
    sequence - The sequence of observations over time. sequence[time] = obs
    """
    n = len(init_probs)
    states = range(n)

    time_state_prob = [[0]*n for _ in sequence]
    for state, prob in enumerate(init_probs):
        time_state_prob[0][state] = prob * emit_probs[state][sequence[0]]

    for t, observed in enum_range(sequence, 1):
        if t % 10000 == 0:
            print(t)
        for state in states:
            obs_prob = emit_probs[state][sequence[t]]
            trans_prob = sum(trans_probs[prev_state][state] 
                    * time_state_prob[t-1][prev_state] 
                    for prev_state in states)
            time_state_prob[t][state] = obs_prob * trans_prob
    return time_state_prob


def backward_prob(trans_probs, emit_probs, sequence):
    """
    For each state at each time step, calculates the probability that the
    given observations after that time step would occur given being in that
    state.
    trans_probs - The transition probabilities between states as 
        a list of lists. trans_prob[source state][dest state]
    emit_probs - The probabilities that each state will produce each 
        observation as a list of lists (or list of dicts).
        emit_probs[state][observation]
    sequence - The sequence of observations over time. sequence[time] = obs
    """
    n = len(trans_probs)
    states = range(n)

    time_state_prob = [[0]*n for _ in sequence]
    for state in states:
        time_state_prob[-1][state] = 1

    for t, observed in enum_range(sequence, -1, 0, -1):
        for state in states:
            time_state_prob[t-1][state] = sum(time_state_prob[t][next_state] 
                    * trans_probs[state][next_state] 
                    * emit_probs[next_state][observed] 
                    for next_state in states)
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

