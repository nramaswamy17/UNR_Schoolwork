import random
def quick_sort(A, p, r):
    if r is None:
        r = len(A) - 1
    if p >= r:
        return
    if p < r:
        q = partition(A, p, r)
        quick_sort(A, p, q-1) # Left half
        quick_sort(A, q, r) # Right half

import random

def partition(A, p, r):
    k = random.randint(p, r)      # random pivot choice
    A[k], A[r] = A[r], A[k]
    x = A[r]
    i = p - 1
    for j in range(p, r):
        if A[j] <= x:
            i += 1
            A[i], A[j] = A[j], A[i]
    A[i+1], A[r] = A[r], A[i+1]
    return i + 1


A = [5, 3, 6, 4, 8]
quick_sort(A, 0, len(A)-1)
print(A)