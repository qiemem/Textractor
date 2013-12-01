import fileinput
import argparse
import random
from collections import defaultdict
import itertools
import numpy as np

def normalized(array):
    return array / np.sum(array)

def distribution(size):
    return normalized(np.random.uniform(0, 1, size=size))

def random_hmm(num_states, num_observables):
    states = range(num_states)
    trans_probs = np.array([distribution(num_states) for s in states])
    init_probs = distribution(num_states)
    emit_probs = np.array([distribution(num_observables) for s in states])
    return HMM(trans_probs, init_probs, emit_probs)

class HMM(object):
    def __init__(self, trans_probs, init_probs, emit_probs):
        self.trans_probs = trans_probs
        self.init_probs = init_probs
        self.emit_probs = emit_probs


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

def forward_prob(trans_probs, init_probs, emit_probs, sequence, normalize = True):
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

    Returns a matrix alpha representing the forward probabilities such that
    alpha[t, s] = P(seq[0],..., seq[t], state[t] = s). If normalize is true, 
    then the values of the alpha[t,:] are normalized on 
    P(seq[t] | seq[0], ..., seq[t-1]) and those normalizing factors are 
    returned along with the matrix.

    To recover the original, do matrix[t,s] * normalizers[:t+1].prod()
    """
    n = len(init_probs)
    states = range(n)

    time_state_prob = np.array([[0.0]*n for _ in sequence])
    time_state_prob[0,:] = init_probs * emit_probs[:, sequence[0]]
    if normalize:
        normalizers = np.array([0.0]*len(sequence))
        normalizers[0] = time_state_prob[0,:].sum()
        time_state_prob[0,:] /= normalizers[0]

    # note that normalizers[t] is P(seq[t] | seq[0], ..., seq[t-1])
    # see http://cs.brown.edu/courses/archive/2006-2007/cs195-5/lectures/lecture33.pdf
    # Use that instead of wikipedia. It has formulas for normalized numbers.
    # Math checks out.

    for t, observed in enum_range(sequence, 1):
        time_state_prob[t,:] = emit_probs[:, sequence[t]] * time_state_prob[t-1,:].dot(trans_probs)
        if normalize:
            normalizers[t] = time_state_prob[t,:].sum()
            time_state_prob[t,:] /= normalizers[t]
    if normalize:
        return time_state_prob, normalizers
    else:
        return time_state_prob


def backward_prob(trans_probs, emit_probs, sequence, normalize = True):
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

    Returns a matrix beta representing the backward probabilities such that
    beta[t, s] = P(seq[t+1],..., seq[T] | state[t] = s). If normalize is true, 
    then the values of the beta[t,:] are normalized on 
    P(seq[t + 1] | seq[0], ..., seq[t]) and those normalizing factors are 
    returned along with the matrix.

    FIXME: I think normalization factors may be off by one. I can't quite tell
    how the normalization should affect beta[T,:].
    See http://cs.brown.edu/courses/archive/2006-2007/cs195-5/lectures/lecture33.pdf
    and http://courses.media.mit.edu/2010fall/mas622j/ProblemSets/ps4/tutorial.pdf

    """
    n = len(trans_probs)
    states = range(n)

    time_state_prob = np.array([[0.0]*n for _ in sequence])
    time_state_prob[-1,:] = 1.0
    if normalize:
        normalizers = np.array([0.0]*len(sequence))
        normalizers[-1] = time_state_prob[-1,:].sum()
        time_state_prob[-1,:] /= normalizers[-1]

    for t, observed in enum_range(sequence, -1, 0, -1):
        time_state_prob[t-1,:] = trans_probs.dot(time_state_prob[t,:] * emit_probs[:,observed])
        if normalize:
            normalizers[t-1] = time_state_prob[t-1,:].sum()
            time_state_prob[t-1,:] /= normalizers[t-1]
    if normalize:
        return time_state_prob, normalizers
    else:
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

