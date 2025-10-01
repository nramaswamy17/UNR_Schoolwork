def bubble_sort(A):
    n = len(A)
    for i in range(n-1):
        print(A)
        for j in reversed(range(1,n-i)):
            if A[j] < A[j-1]:
                tmp = A[j]
                A[j] = A[j-1]
                A[j-1] = tmp
    print(A)
A = [5, 3, 6, 4, 8]
bubble_sort(A)