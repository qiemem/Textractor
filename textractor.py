import fileinput

def make_tuples(tuple_size, sentences):
    for sentence in sentences:
        words = sentence.split()
        for i, w in enumerate(words[:-(tuple_size - 1)]):
            yield tuple(words[i:i+tuple_size])

if __name__ == '__main__':
    for t in make_tuples(3, fileinput.input()):
        print(t)

