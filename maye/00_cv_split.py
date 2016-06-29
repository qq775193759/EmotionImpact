import xml.etree.ElementTree as ET
from collections import defaultdict
from random import shuffle


a_map = {}
v_map = {}

with open('ACCEDEranking.txt', 'r') as fin:
    for line in fin:
        l = line.strip().split('\t')
        if l[0] == 'id':
            continue
        id = int(l[0])
        a = float(l[5])
        v = float(l[4])
        a_map[id] = a
        v_map[id] = v


tree = ET.parse('ACCEDEdescription.xml')
root = tree.getroot()

l = []
d_count = defaultdict(int)
movies_set = {}
id_set = {}
counts = [0, 0, 0, 0, 0]    # 5-folds cv
arousals = [0, 0, 0, 0, 0]
valences = [0, 0, 0, 0, 0]
current = 0
MAX = 1960

for child in root:
    d = {'id': int(child[0].text), 'name': child[1].text, 'movie': child[3].text}
    d_count[child[3].text] += 1
    l.append(d)

ll = list(d_count.items())
shuffle(ll)


for item in ll:
    c = item[1]
    movie = item[0]
    if counts[current] + c > MAX:
        if MAX - counts[current] > counts[current] + c - MAX:
            movies_set[movie] = current
            counts[current] += c
            if current < 4:
                current += 1
        else:
            if current < 4:
                current += 1
            movies_set[movie] = current
            counts[current] += c
    else:
        movies_set[movie] = current
        counts[current] += c

for i in l:
    id = i['id']
    s = movies_set[i['movie']]
    arousals[s] += a_map[id]
    valences[s] += v_map[id]
    id_set[id] = s

for i in range(5):
    arousals[i] /= counts[i]
    valences[i] /= counts[i]

with open('CVsets.txt', "w") as fout, open('annotations0.txt', "w") as f0,\
        open('annotations1.txt', "w") as f1, open('annotations2.txt', "w") as f2,\
        open('annotations3.txt', "w") as f3, open('annotations4.txt', "w") as f4:
    for i in range(9800):
        fout.write(str(i) + '\t' + str(id_set[i]) + '\n')
        if id_set[i] == 0:
            f0.write(str(i) + '\t' + str(v_map[i]) + '\t' + str(a_map[i]) + '\n')
        elif id_set[i] == 1:
            f1.write(str(i) + '\t' + str(v_map[i]) + '\t' + str(a_map[i]) + '\n')
        elif id_set[i] == 2:
            f2.write(str(i) + '\t' + str(v_map[i]) + '\t' + str(a_map[i]) + '\n')
        elif id_set[i] == 3:
            f3.write(str(i) + '\t' + str(v_map[i]) + '\t' + str(a_map[i]) + '\n')
        elif id_set[i] == 4:
            f4.write(str(i) + '\t' + str(v_map[i]) + '\t' + str(a_map[i]) + '\n')
        else:
            assert False
