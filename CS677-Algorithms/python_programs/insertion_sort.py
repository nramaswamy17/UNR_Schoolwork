def insertion_sort(A):
    n = len(A)
    for j in range(1,n):
        print(A)
        key = A[j] # Set the desired sort element
        i  = j - 1
        print("  i =", i, "A[i] =", A[i], "key =", key)
        while i >= 0 and A[i] > key:
            A[i+1] = A[i]
            i = i-1
        A[i+1] = key
    print(A)
    
A = [5, 3, 6, 4, 8]
insertion_sort(A)
print(A)