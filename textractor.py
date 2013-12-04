import fileinput
import argparse
import random
from collections import defaultdict
import itertools
import numpy as np
import sys
from preprocess_text import stem, isStopWord

def normalized(array):
    return array / np.sum(array)

def distribution(size):
    return normalized(np.array([random.random() for i in xrange(size)]))

def make_tuples(tuple_size, seqs):
    for seq in seqs:

        for i, w in enumerate(seq[:-(tuple_size - 1)]):

            yield tuple(seq[i:i+tuple_size])

def random_hmm(num_states, num_observables,
        trans_probs = None, init_probs = None, emit_probs = None):
    states = range(num_states)
    if not isinstance(trans_probs, np.ndarray): 
        trans_probs = np.array([distribution(num_states) for s in states])
    if not isinstance(init_probs, np.ndarray):
        init_probs = distribution(num_states)
    if not isinstance(emit_probs, np.ndarray): 
        emit_probs = np.array([distribution(num_observables) for s in states])
    return HMM(trans_probs, init_probs, emit_probs)

def make_modded_cooccurrence(num_states,num_observables, sequences):
    states = range(num_states)
    emit_probs = np.zeros((len(states), num_observables))
    for w1, w2 in make_tuples(2, sequences):
        emit_probs[hash(w1)%num_states, word_codes[w2]] += 1
        emit_probs[hash(w2)%num_states, word_codes[w1]] += 1
    for i in states:
        emit_probs[i]=normalized(emit_probs[i])
    return emit_probs

def weighted_random(weights):
    r = random.random()
    for i,w in enumerate(weights):
        r -= w
        if r < 0:
            return i

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

    def gen_seq(self, length):
        seq = np.zeros(length, dtype = np.int)
        state = weighted_random(self.init_probs)
        for i in xrange(length):
            seq[i] = weighted_random(self.emit_probs[state])
            state = weighted_random(self.trans_probs[state])
        return seq

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
        """
        Calculates the probability of reaching each state at each timestep.
        """
        return normed_forward_probs * normed_backward_probs

    def expected_trans(self, sequence, normed_forward_probs, normed_backward_probs, normalizers):
        """
        Calculates the probability of transitioning from each state to each 
        other state at each timestep.
        """
        states = range(len(self.init_probs))
        result = np.zeros((len(sequence)-1, len(states), len(states)))
        for t, word in enum_range(sequence,0,-1,1):
            next_word = sequence[t+1]
            result[t] = normed_forward_probs[t,:,np.newaxis] 
            result[t] *= self.emit_probs[:,next_word] 
            result[t] *= self.trans_probs 
            result[t] *= normed_backward_probs[t+1] / normalizers[t+1]
        return result

    def improve(self, sequences):
        """
        Runs a single iteration of EM on the given sequences. The improved HMM
        is returned; this one is unaffected.
        """
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

            new_init_probs += state_probs[0]
            
            trans_probs_num += expected_trans.sum(0)
            trans_probs_denom += state_probs[:-1].sum(0)

            emit_probs_denom += state_probs.sum(0)
            for word in set(seq):
                emit_probs_num[:,word] += state_probs[seq==word].sum(0)
            nll -= np.log(normalizers).sum()

# trans_probs_denom can get zeros in it if a node becomes unreachable. That
# creates NaNs everywhere. So, we smooth just a tiny amount.
# Note that trans_probs_denom is the expected number of visits to each node.
        trans_probs_denom += 1.0
        trans_probs_num += 1.0 / len(trans_probs_denom)
        new_trans_probs = (trans_probs_num.transpose() / trans_probs_denom).transpose()
        emit_probs_num += 1.0 / len(emit_probs_num[0])
        emit_probs_denom += 1
        new_emit_probs = (emit_probs_num.transpose() / emit_probs_denom).transpose()
        return HMM(new_trans_probs, new_init_probs / len(sequences), new_emit_probs), nll

def maximize_expectation(hmm, sequences, max_iters = 10000, nll_percent = 0.00001, print_nll = False):
    last_l = np.inf
    for i in xrange(max_iters):
        #hmm, l = hmm.improve(random.sample(sequences, 10000))
        hmm, l = hmm.improve(sequences)
        if print_nll:
            log(l)
        if nll_percent > (1 - l / last_l) and l <= last_l:
            return hmm
        last_l = l
        if np.isnan(l):
            log('NaNs detected!!!')
            return hmm
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

def load_sequences(filename, do_stem = False):
    """
    Returns (seqs, words, word_codes, coded_seqs) .`words` is never stemmed.
    All others use stemming if do_stem is true.
    """
    seqs = (seq.split() for seq in fileinput.input(filename))
    seqs = [seq for seq in seqs if len(seq)>0]

    words = list({w for seq in seqs for w in seq})
    if do_stem:
        seqs = [[stem(w) for w in seq if not isStopWord(w)] for seq in seqs]
        stemmed_words = list({w for seq in seqs for w in seq})
    else:
        stemmed_words = words
    word_codes = {w: i for i,w in enumerate(stemmed_words)}
    coded_seqs = [np.array([word_codes[w] for w in seq]) for seq in seqs]
    return seqs, words, word_codes, coded_seqs
    

def log(string):
    sys.stderr.write(str(string))
    sys.stderr.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Given a bunch of sentences, outputs feature vectors of the words')
    parser.add_argument('-n', default=100, type=int, 
            help='Length of feature vectors (default is 100)')
    parser.add_argument('-f', default='-', type=str, metavar='filename',
            help='File containing sentences to process (defaults to stdin)')
    parser.add_argument('-s', default=False, action='store_true',
            help='Run HMM on stemmed words')
    parser.add_argument('-i', default=10000, type=int,
            help='Maximum number of iterations of EM to do')
    parser.add_argument('--seed', default=False, action='store_true',
            help='Emission probabilities are seeded with a modded co-occurrence matrix')

    args = parser.parse_args()

    log('Reading sequences')
    seqs, words, word_codes, coded_seqs = load_sequences(args.f, args.s)

    log('Generating initial HMM')
    if args.seed:
        emit_probs = make_modded_cooccurrence(args.n,len(word_codes), seqs)
        init_hmm = random_hmm(args.n, len(words), emit_probs = emit_probs)
    else:
        init_hmm = random_hmm(args.n, len(words))
    log('Running EM')
    final_hmm = maximize_expectation(init_hmm, coded_seqs, max_iters = args.i, print_nll = True)
    log('Writing results')
    for w in words:
        if not args.s or not isStopWord(w):
            i = word_codes[stem(w)] if args.s else word_codes[w]
            print(w + ' ' + ' '.join(str(x) for x in final_hmm.emit_probs[:, i]))

