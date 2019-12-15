from operator import itemgetter

unigram_count = {}

with open("model.txt", "r") as f:
    t = f.readline().strip().split()
    unique_words = int(t[0])
    total_words = int(t[1])

    for i in range(unique_words):
        t = f.readline().strip().split()
        w = t[1]
        count = int(t[2])

        unigram_count[w] = count

def edits1(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def known(words):
    return set((w, unigram_count[w]) for w in words if w in unigram_count)

total_misspelled_words = 0 # Total number of misspelled words (defined as all misspelled variants of a single word).
correct_corrections = 0 # The number of correct suggestions that were made.
window_size = 3 # Prediction window size.

with open("missp.dat", "r") as f:
    expected_word = None
    misspelled_word = None
    i=0
    for line in f:
        i+=1
        if i%10 == 0:
            print("\nlines processed thus far", i)
            print("misspelled words thus far", total_misspelled_words)
            print("correct corrections thus far", correct_corrections)
        line = line.strip()
        if line[0] == "$":
            expected_word = line[1:]
            continue
        misspelled_word = line[:]
        total_misspelled_words += 1

        perm = edits2(misspelled_word) # permutations
        s = known(perm)
        s = list(s)
        s.sort(key=itemgetter(1), reverse=True)
        k=s[:window_size]
        d=[w for (w,c) in k]
        if expected_word in d:
            correct_corrections += 1

print("total:", total_misspelled_words)
print("corrected:", correct_corrections)
print("relative:", correct_corrections * 100 / total_misspelled_words)
