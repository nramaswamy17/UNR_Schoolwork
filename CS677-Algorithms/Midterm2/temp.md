RANDOMIZED-SELECT(A, 0, 6, 3)
A = [7, 2, 9, 4, 5, 6, 1]
lo = 0, hi = 6, i = 3


## First Recursive Call

**Step 1: Check base case**
```
if lo = hi? → 0 ≠ 6, so continue
```

**Step 2: Find median**
```
mid ← Median(A, 0, 6)
```
The median of `[7, 2, 9, 4, 5, 6, 1]` is **mid = 5**

**Step 3: Partition around median**
```
q ← Partition(A, 0, 6, 5)
```
Rearrange so elements ≤ 5 are on left, elements > 5 are on right:
```
Before: [7, 2, 9, 4, 5, 6, 1]
After:  [2, 4, 1, 5, 7, 9, 6]
         ^^^^^^^  ^  ^^^^^^^
         ≤ mid  pivot  > mid
```

The pivot (5) ends up at index `q = 3`.

**Step 4: Calculate k**
```
k ← q - lo + 1 = 3 - 0 + 1 = 4
```
The value 5 is the **4th smallest** element in our current range [0, 6].

**Step 5: Compare i and k**
```
i = 3, k = 4
i < k, so the answer is in the LEFT half
```

Recurse on the left: `[2, 4, 1]`, still looking for the 3rd smallest.

---

## Second Recursive Call
```
RANDOMIZED-SELECT(A, 0, 2, 3)
A = [2, 4, 1, 5, 7, 9, 6]
Current range: [lo=0 to hi=2] → [2, 4, 1]
lo = 0, hi = 2, i = 3
```

**Step 1: Check base case**
```
if lo = hi? → 0 ≠ 2, so continue
```

**Step 2: Find median**
```
mid ← Median([2, 4, 1], 0, 2)
```
The median of `[2, 4, 1]` is **mid = 2**

**Step 3: Partition around median**
```
q ← Partition(A, 0, 2, 2)
```
Rearrange:
```
Before: [2, 4, 1]
After:  [1, 2, 4]
         ^  ^  ^
        <mid =mid >mid
```

The pivot (2) ends up at index `q = 1`.

**Step 4: Calculate k**
```
k ← q - lo + 1 = 1 - 0 + 1 = 2
```
The value 2 is the **2nd smallest** in the range [0, 2].

**Step 5: Compare i and k**
```
i = 3, k = 2
i > k, so the answer is in the RIGHT half
```

Recurse on the right: `[4]`, with adjusted i:
- New lo = q + 1 = 2
- New hi = 2
- New i = i - k = 3 - 2 = **1**

---

## Third Recursive Call
```
RANDOMIZED-SELECT(A, 2, 2, 1)
Current range: [lo=2 to hi=2] → [4]
lo = 2, hi = 2, i = 1
```

**Step 1: Check base case**
```
if lo = hi? → 2 = 2, YES!
return A[2] = 4