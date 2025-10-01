def selection_sort(A):
    n = len(A)
    for j in range(n-1):
        print(A)
        smallest = j
        for i in range(j+1, n):
            if A[i] < A[smallest]:
                smallest = i
        tmp = A[smallest]
        A[smallest] = A[j]
        A[j] = tmp
    print(A)


A = [5, 3, 6, 4, 8]
selection_sort(A)
