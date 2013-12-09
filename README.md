Textractor
==========

Constructs a probabilistic graphical model based on a given set of sentences in order to construct a feature vector for each word.

To accomplish this, the script constructs a hidden Markov model with the given number of hidden states.
It then runs expectation maximatization on the HMM for the given number of iterations.
Finally, it outputs the emission probabilities for each word.

Several modifications of the core algorithm are available.
The data can be pre-processed with stemming and stop word removal.
This reduces the number of observable states of the HMM as well as the length of the corpus, speeding up the EM iterations.
However, it appears to hurt results.

The emission probabilities can be seeded with an arbitrarily partitioned co-occurrence matrix.
This improves results in general.

The Porter Stemmer implementation is taken from [NLTK](http://nltk.org/).

Usage
---

    usage: textractor.py [-h] [-n N] [-f filename] [-s] [-i I] [--seed]
    					 [--out file prefix]

    Given a bunch of sentences, outputs feature vectors of the words

    optional arguments:
      -h, --help         show this help message and exit
      -n N               Length of feature vectors (default is 100)
      -f filename        File containing sentences to process (defaults to stdin)
      -s                 Stem words and remove stop words.
      -i I               Maximum number of iterations of EM to do.
      --seed             Seed emission probabilities with a partitioned co-occurrence matrix
      --out file prefix  Output intermittent data in files prefixed with this
    					 argument. By default, final results are printed to
    					 stdout.

For example:

    python textractor.py -n 20 -i r -f dataset.txt > output.txt

Note that it runs much faster with pypy.
