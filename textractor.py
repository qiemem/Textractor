import fileinput
import os
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

def make_modded_cooccurrence(num_states, num_observables, coded_seqs, smoothing = 1):
    states = range(num_states)
    emit_probs = np.ones((len(states), num_observables)) * smoothing
    for w1, w2 in make_tuples(2, coded_seqs):
        emit_probs[w1 % num_states, w2] += 1
        emit_probs[w2 % num_states, w1] += 1
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

        Returns an array alpha representing the forward probabilities such that
        alpha[t, s] = P(seq[0],..., seq[t], state[t] = s). If normalize is true, 
        then the values of the alpha[t] are normalized on 
        P(seq[t] | seq[0], ..., seq[t-1]) and those normalizing factors are 
        returned along with the matrix.

        To recover the original, do matrix[t,s] * normalizers[:t+1].prod()
        """
        n = len(self.init_probs)
        states = range(n)

        time_state_probs = self.emit_probs[:, sequence].T
        time_state_probs[0] *= self.init_probs
        if normalize:
            normalizers = np.zeros(sequence.shape)
            normalizers[0] = time_state_probs[0].sum()
            time_state_probs[0] /= normalizers[0]

        # note that normalizers[t] is P(seq[t] | seq[0], ..., seq[t-1])
        # see http://cs.brown.edu/courses/archive/2006-2007/cs195-5/lectures/lecture33.pdf
        # Use that instead of wikipedia. It has formulas for normalized numbers.
        # Math checks out.

        for t in xrange(1,len(sequence)):
            time_state_probs[t] *= time_state_probs[t-1].dot(self.trans_probs)
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

        Returns an array beta representing the backward probabilities such that
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
        Resulting array is gamma[timestep, state] = probabilities
        """
        return normed_forward_probs * normed_backward_probs

    def expected_trans(self, sequence, normed_forward_probs, normed_backward_probs, normalizers):
        """
        Calculates the probability of transitioning from each state to each 
        other state at each timestep.
        Resulting array is xi[timestep, state, state] -> probabilities
        xi is (timesteps - 1 X states X states)
        """
        states = range(len(self.init_probs))
        result = np.ones((len(sequence)-1, len(states), len(states)))
        result *= normed_forward_probs[:-1,:,np.newaxis] 
        result *= self.emit_probs[:,sequence[1:]].T[:,np.newaxis]
        result *= self.trans_probs
        result *= (normed_backward_probs[1:] / normalizers[1:,np.newaxis])[:,np.newaxis]
        return result

    def nll(self, sequences):
        nll = 0
        for seq in sequences:
            forward, normalizers = self.forward_probs(seq, True)
            nll -= np.log(normalizers).sum()
        return nll


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

            new_init_probs += state_probs[0]
            emit_probs_denom += state_probs.sum(0)
            for t, word in enumerate(seq):
                emit_probs_num[:,word] += state_probs[t]
            
# Besides just being reasonable to avoid training transition probabilities on 
# sequences with only one element, pypy barfs on summing empty, 
# multidimensional arrays.
            if len(seq) > 1:
                expected_trans = self.expected_trans(seq, forward, backward, normalizers)
                trans_probs_num += expected_trans.sum(0)
                trans_probs_denom += state_probs[:-1].sum(0)

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

def em(hmm, sequences):
    while True:
        hmm, nll = hmm.improve(sequences)
        yield nll, hmm

def maximize_expectation(hmm, sequences, max_iters = 10000, print_nll = False, out_func=None):
    em_iter = em(hmm, sequences)
    if out_func:
        last_fn = out_func(hmm, 0)
    for i in xrange(max_iters):
        nll, next_hmm = next(em_iter)
        if out_func:
            os.rename(last_fn, last_fn[:-3]+'_'+str(nll)+'.txt')
        if print_nll:
            log(nll)
        hmm = next_hmm
        if out_func:
            last_fn = out_func(hmm, i+1)
    if out_func:
        os.rename(last_fn, last_fn[:-3]+'_'+str(hmm.nll(sequences))+'.txt')
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
    split_lines = [seq.split() for seq in fileinput.input(filename)]
    words = list({w for seq in split_lines for w in seq})
    if do_stem:
        seqs = ([stem(w) for w in seq if not isStopWord(w)] for seq in split_lines)
        stemmed_words = list({stem(w) for w in words if not isStopWord(w)})
    else:
        seqs = split_lines
        stemmed_words = words
    word_codes = {w: i for i,w in enumerate(stemmed_words)}
    coded_seqs = [np.array([word_codes[w] for w in seq]) for seq in seqs if len(seq) > 0]
    return words, word_codes, coded_seqs

def log(string):
    sys.stderr.write(str(string))
    sys.stderr.write('\n')

def output_results(prefix, states, stemmed, iterations, seed, words, word_codes, hmm, i):
    filename ='{}_states-{}_stemmed-{}_iters-{}_seed-{}_{}.txt'.format(
            prefix, states, stemmed, iterations, seed, i)
    with open(filename , 'w') as out:
        for w in words:
            if not stemmed or not isStopWord(w):
                i = word_codes[stem(w)] if stemmed else word_codes[w]
                out.write(w + ' ' + ' '.join(str(x) for x in hmm.emit_probs[:, i]))
                out.write('\n')
    return filename

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Given a bunch of sentences, outputs feature vectors of the words')
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
    parser.add_argument('--out', default=None, type=str, metavar='file prefix',
            help='Output intermittent data in files prefixed with this argument. By default, final results are printed to stdout.')

    args = parser.parse_args()

    log('Reading sequences')
    words, word_codes, coded_seqs = load_sequences(args.f, args.s)

    log('{} words, {} sequences, {} observables'.format(len(words), len(coded_seqs), len(word_codes)))
    log('Generating initial HMM')
    if args.seed:
        emit_probs = make_modded_cooccurrence(args.n, len(word_codes), coded_seqs)
        init_hmm = random_hmm(args.n, len(words), emit_probs = emit_probs)
    else:
        init_hmm = random_hmm(args.n, len(words))
    log('Running EM')
    out_func = lambda hmm, i : output_results(args.out, args.n, args.s, args.i, args.seed, words, word_codes, hmm, i) if args.out else None
    final_hmm = maximize_expectation(init_hmm, coded_seqs, max_iters = args.i, print_nll = True, out_func = out_func)
    if not args.out:
        log('Writing results')
        for w in words:
            if not args.s or not isStopWord(w):
                i = word_codes[stem(w)] if args.s else word_codes[w]
                print(w + ' ' + ' '.join(str(x) for x in final_hmm.emit_probs[:, i]))

