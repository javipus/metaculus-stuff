from functools import reduce
from numpy import ceil
seats = 3

# only valid ones
ballots = [
    ['a', 'b'],
    ['c', 'b', 'a'],
    ['b', 'd']
]

candidates = set(reduce(lambda x,y: x+y, ballots))
points = {c: sum(b[0] == c for b in ballots) for c in candidates}
quota = int(ceil((len(ballots) + 1) / (seats + 1)))
n_elected = 0

while True:
    elected = {c: points[c] >= quota for c in candidates}
    n_elected = sum(elected.values())
    
    if n_elected >= seats: break

    surplus = {c: max(0, points[c] - quota*elected[c]) for c in candidates}
    transfer = {c: surplus[c] / points[c] for c in candidates}

    for b in ballots:
        if len(b) > 1:
            points[b[1]] += transfer[b[0]]
            points[b[0]] -= 1
    
    if all(p < quota for p in points.values()):
        # TODO implement this
        drop_lowest()