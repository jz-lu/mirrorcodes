from itertools import *

con = [[1,2,3],[0,4],[0,5],[0,6],[1,7],[2,7],[3,7],[4,5,6]]

total = 0
for i in permutations(range(8)):
    valid = True
    for j in range(7):
        if i[j] > i[j+1] and i[j] not in con[i[j+1]]:
            valid = False
    if not valid:
        continue
    count = 0
    where = [i.index(j) for j in range(8)]
    before = [0] * 8
    for j in range(8):
        for k in con[i[j]]:
            if j > where[k]:
                before[j] += 1
        if (len(con[i[j]]), before[j]) in [(3, 0), (3, 3), (2, 0), (2, 2)]:
            count += 2
        else:
            count += 1
    if count >= 13:
        continue
    mustz = [[
        where[4] < where[1] < where[0],
        where[0] < where[1] < where[4] or where[7] < where[4] < where[1],
        where[1] < where[4] < where[7]], [
        where[5] < where[2] < where[0],
        where[0] < where[2] < where[5] or where[7] < where[5] < where[2],
        where[2] < where[5] < where[7]], [
        where[6] < where[3] < where[0],
        where[0] < where[3] < where[6] or where[7] < where[6] < where[3],
        where[3] < where[6] < where[7]]]
    non_FT = False
    for j in product(range(3), repeat = 3):
        if sum(j) != 3:
            continue
        if sum([1 if mustz[k][j[k]] else 0 for k in range(3)]) >= 2:
            non_FT = True
            break
    if non_FT:
        continue
    print(i)
    total += 1
print(total)
