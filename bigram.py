import torch
import re

def ctoi(c):
    return ctoimap[c]

def itoc(i):
    return itocmap[i]

words = open("sample.txt").read().split()
alphabet = set(c for word in words for c in word)
alphabet.add('<')
alphabet.add('>')
model = torch.zeros((len(alphabet),len(alphabet),), dtype=torch.double)
ctoimap = { c:i for (i,c) in enumerate(alphabet)}
itocmap = { i:c for (i,c) in enumerate(alphabet)}

print ("Constructing model")
for word in words:
    word = re.sub(r'[^a-zA-Z]+', '', word)
    if len(word) == 0:
        continue
    word = "<" + word + ">"
    bigrams =  (list(zip(word, word[1:])))
    for (a,b) in bigrams:
        #print (f'{ctoi(a), ctoi(b)}')
        model[ctoi(a),ctoi(b)] += 1

model = torch.nn.functional.normalize(model, dim=1)

print ("Generating words")
MINLEN = 4
nwords = 0
while nwords < 10:
    char = '<'
    res=char
    while char != '>':
        row = model[ctoi(char)]
        idx = torch.multinomial(row, num_samples=1, replacement=True)
        char = itoc(idx.item())
        res += char

    if len(res) >= MINLEN + 2:
        print(res)
        nwords += 1

