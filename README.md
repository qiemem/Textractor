Textractor
==========

Constructs a probabilistic graphical model based on a given set of sentences in order to construct a feature vector for each word.

Currently, the script uses an incredibly simple mechanism to determine
similarity between concepts. It uses a twist on a co-occurrence matrix.
A co-occurrence matrix has each row for each word and a column for each word.
Then, cell i,j is equal to the number of times word i and word j appear 
together. However, we want to limit the length of each word's vector. So, each
row is only 100 dimensions. Cell i,j is the number of times that word i 
appears next to a word who's hash value mod 100 is j.

Usage
---

    usage: textractor.py [-h] [-n N] [-f filename]

    Given a bunch of sentences, outputs feature vectors of the words

    optional arguments:
      -h, --help   show this help message and exit
      -n N         Length of feature vectors (default is 100)
      -f filename  File containing sentences to process (defaults to stdin)

For example:

    python textractor.py -n 20 -f dataset.txt > output.txt

Note that it runs much faster with pypy.
