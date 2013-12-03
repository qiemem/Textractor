import fileinput
import argparse
import random
from collections import defaultdict
import itertools
import numpy as np
import sys

def normalized(array):
    return array / np.sum(array)

def distribution(size):
    return normalized(np.array([random.random() for i in xrange(size)]))

def random_hmm(num_states, num_observables):
    states = range(num_states)
    trans_probs = np.array([distribution(num_states) for s in states])
    init_probs = distribution(num_states)
    emit_probs = np.array([distribution(num_observables) for s in states])
    return HMM(trans_probs, init_probs, emit_probs)

class HMM(object):
    def __init__(self, trans_probs, init_probs, emit_probs):
        """
        trans_probs - The transition probabilities between states as 
            an array of arrays. trans_prob[source state, dest state]
        init_probs - The probabilities of beginning in any given state as vector
            indexed by state.
        emit_probs - The probabilities that each state will produce each 
            observation as an array of arrays.
            emit_probs[state,observation]
        """
        self.trans_probs = trans_probs
        self.init_probs = init_probs
        self.emit_probs = emit_probs

    def forward_probs(self, sequence, normalize = False):
        """
        For each state at each time step, calculates the probability that the hmm
        would produce the given sequence of observations up to that time step and
        land in that state.
        sequence - The sequence of observations over time. sequence[time] = obs

        Returns a matrix alpha representing the forward probabilities such that
        alpha[t, s] = P(seq[0],..., seq[t], state[t] = s). If normalize is true, 
        then the values of the alpha[t] are normalized on 
        P(seq[t] | seq[0], ..., seq[t-1]) and those normalizing factors are 
        returned along with the matrix.

        To recover the original, do matrix[t,s] * normalizers[:t+1].prod()
        """
        n = len(self.init_probs)
        states = range(n)

        time_state_probs = np.zeros((len(sequence), n))
        time_state_probs[0] = self.init_probs * self.emit_probs[:, sequence[0]]
        if normalize:
            normalizers = np.zeros(sequence.shape)
            normalizers[0] = time_state_probs[0].sum()
            time_state_probs[0] /= normalizers[0]

        # note that normalizers[t] is P(seq[t] | seq[0], ..., seq[t-1])
        # see http://cs.brown.edu/courses/archive/2006-2007/cs195-5/lectures/lecture33.pdf
        # Use that instead of wikipedia. It has formulas for normalized numbers.
        # Math checks out.

        for t, observed in enum_range(sequence, 1):
            time_state_probs[t] = self.emit_probs[:, sequence[t]] * time_state_probs[t-1].dot(self.trans_probs)
            if normalize:
                normalizers[t] = time_state_probs[t].sum()
                time_state_probs[t] /= normalizers[t]
        if normalize:
            return time_state_probs, normalizers
        else:
            return time_state_probs

    def backward_probs(self, sequence, normalizers = None):
        """
        For each state at each time step, calculates the probability that the
        given observations after that time step would occur given being in that
        state.
        sequence - The sequence of observations over time. sequence[time] = obs
        normalizers - If given, normalizes each value of the result matrix
            so normed_beta[t,s] = beta[t, s] / normalizers[t+1:].prod()

        Returns a matrix beta representing the backward probabilities such that
        beta[t, s] = P(seq[t+1],..., seq[T] | state[t] = s).
        """
        n = len(self.trans_probs)
        states = range(n)

        time_state_probs = np.zeros((len(sequence), n))
        time_state_probs[-1] = 1.0

        for t, observed in enum_range(sequence, -1, 0, -1):
            time_state_probs[t-1] = self.trans_probs.dot(time_state_probs[t] * self.emit_probs[:,observed])
            if isinstance(normalizers, np.ndarray):
                time_state_probs[t-1] /= normalizers[t]
        return time_state_probs

    def state_probs(self, normed_forward_probs, normed_backward_probs):
        return normed_forward_probs * normed_backward_probs

    def expected_trans(self, sequence, normed_forward_probs, normed_backward_probs, normalizers):
        states = range(len(self.init_probs))
        result = np.zeros((len(sequence)-1, len(states), len(states)))
        for t, word in enum_range(sequence,0,-1,1):
            next_word = sequence[t+1]
            for source_state in states:
                for dest_state in states:
                    result[t, source_state, dest_state] = normed_forward_probs[t,source_state] * self.trans_probs[source_state, dest_state] * self.emit_probs[dest_state, next_word] * normed_backward_probs[t+1, dest_state] / normalizers[t+1]
        return result

    def improve(self, sequences):
        num_seqs = len(sequences)
        new_init_probs = np.zeros(self.init_probs.shape)
        trans_probs_num = np.zeros(self.trans_probs.shape)
        trans_probs_denom = np.zeros(self.init_probs.shape)
        emit_probs_num = np.zeros(self.emit_probs.shape)
        emit_probs_denom = np.zeros(self.emit_probs[:,0].shape)

        nll = 0 # negative log likelihood

        for seq in sequences:
            forward, normalizers = self.forward_probs(seq, True)
            backward = self.backward_probs(seq, normalizers)
            state_probs = self.state_probs(forward, backward)
            expected_trans = self.expected_trans(seq, forward, backward, normalizers)

            new_init_probs += state_probs[1]
            
            trans_probs_num += expected_trans.sum(0)
            trans_probs_denom += state_probs[:-1].sum(0)

            emit_probs_denom += state_probs.sum(0)
            for word in xrange(len(self.emit_probs[0])):
                emit_probs_num[:,word] += state_probs[seq==word].sum(0)
            nll -= np.log(normalizers).sum()

        new_trans_probs = (trans_probs_num.transpose() / trans_probs_denom).transpose()
        new_emit_probs = (emit_probs_num.transpose() / emit_probs_denom).transpose()
        return HMM(new_trans_probs, new_init_probs, new_emit_probs), nll

def maximize_expectation(hmm, sequences, max_iters = 10000, nll_percent = 0.00001, print_nll = False):
    last_l = np.inf
    for i in xrange(max_iters):
        hmm, l = hmm.improve(sequences)
        if print_nll:
            log(l)
        if nll_percent > (1 - l / last_l) and l <= last_l:
            return hmm
        last_l = l
    return hmm

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

def log(string):
    sys.stderr.write(str(string))
    sys.stderr.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Given a bunch of sentences, outputs feature vectors of the words')
    parser.add_argument('-n', default=100, type=int, 
            help='Length of feature vectors (default is 100)')
    parser.add_argument('-f', default='-', type=str, metavar='filename',
            help='File containing sentences to process (defaults to stdin)')

    args = parser.parse_args()

    log('Reading sequences')
    seqs = [seq.split() for seq in fileinput.input(args.f)]
    words = list({w for seq in seqs for w in seq})
    log('Coding sequences')
    word_codes = {w: i for i,w in enumerate(words)}
    coded_seqs = [np.array([word_codes[w] for w in seq]) for seq in seqs]
    log('Generating initial HMM')
    init_hmm = random_hmm(args.n, len(words))
    log('Running EM')
    final_hmm = maximize_expectation(init_hmm, coded_seqs, print_nll = True)

    for i, w in enumerate(words):
        print(w + ' ' + ' '.join(str(x) for x in final_hmm.emit_probs[:, i]))

