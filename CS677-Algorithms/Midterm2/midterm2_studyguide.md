# Midterm 2 Study Guide
## CS 477/677 - Analysis of Algorithms

---

## Topics Covered

- Randomized Quicksort
- Probability Background  
- The Selection Problem
- Sorting in Linear Time (Counting Sort, Radix Sort, Bucket Sort)
- Heaps (Max-Heapify, Build-Max-Heap, Heapsort, Priority Queues)
- Augmenting Data Structures (Red-Black Trees, OS-Trees, Interval Trees)

---

## 1. Randomized Quicksort

### Overview
Randomized algorithms use random-number generators to determine behavior. For Quicksort, randomization prevents consistently bad performance on specific inputs.

### Key Concepts

**Randomized Algorithm**: Behavior determined partly by random number generator. No input can consistently elicit worst-case behavior. Worst case occurs only with "unlucky" random numbers.

**RANDOM(a, b)**: Returns integer r where a ≤ r ≤ b, each value equally likely.

**RANDOMIZED-PARTITION**: 
- Exchange A[p] with randomly chosen element from A[p...r]
- Then call PARTITION on the modified array

### Partition Methods

**PARTITION2** (used in analysis):
- Chooses **last element** A[r] as pivot
- Creates subarrays: A[p..q-1] ≤ pivot, A[q+1..r] > pivot
- **Important**: Pivot at position q is NOT included in recursive calls
- Running time: Θ(n)

**Loop Invariant for PARTITION2**:
- A[p..i] contains elements ≤ pivot
- A[i+1..j-1] contains elements > pivot  
- A[r] = pivot
- A[j..r-1] are elements not yet examined

### Analysis of Randomized Quicksort

- Running time dominated by PARTITION calls
- PARTITION called **at most n times** (each selects a pivot never used again)
- Key insight: Count total comparisons across ALL PARTITION calls
- Each pair of elements compared **at most once** (only to pivot)
- Expected number of comparisons: **O(n lg n)**
- **Expected running time: O(n lg n)**

### When Elements Are Compared

Rename elements as z₁, z₂, ..., zₙ (i-th smallest).

Elements zᵢ and zⱼ are compared if and only if:
- zᵢ or zⱼ is chosen as pivot **before** any element between them

Probability that zᵢ and zⱼ are compared: **2/(j - i + 1)**

---

## 2. Probability Background

### Random Variables

**Random Variable X**: Function from sample space S to real numbers. Associates a real number with each outcome.

**Expected Value**: E[X] = Σₓ x·Pr{X = x}
- "Average" over all possible values
- Example: Fair dice has E[X] = (1+2+3+4+5+6)/6 = 3.5

### Indicator Random Variables

**Definition**: For event A, indicator variable I{A} is:
```
I{A} = 1 if A occurs
I{A} = 0 if A does not occur
```

**Key Property**: **E[I{A}] = Pr{A}**

**Proof**:
```
E[I{A}] = 1·Pr{A} + 0·Pr{Ā} = Pr{A}
```

**Usage**: Extensively used in Quicksort analysis to count comparisons.

---

## 3. The Selection Problem

### Problem Statement
Select the **i-th smallest element** from n distinct numbers (the element larger than exactly i-1 others).

### Order Statistics

- **Minimum**: i = 1, requires **n-1 comparisons** (optimal)
- **Maximum**: i = n, requires **n-1 comparisons** (optimal)
- **Median**: 
  - If n is odd: i = (n+1)/2 (unique)
  - If n is even: i = n/2 (lower median) or n/2+1 (upper median)

### Simultaneous Min and Max

**Naive approach**: Find min (n-1 comparisons) + find max (n-1 comparisons) = **2n-2 comparisons**

**Better approach** (process in pairs): **3n/2 comparisons**
1. Compare elements in pairs
2. Compare smaller element with current min
3. Compare larger element with current max
4. Result: 3 comparisons for every 2 elements

**Analysis**:
- n odd: 3(n-1)/2 comparisons
- n even: 1 + 3(n-2)/2 = 3n/2 - 2 comparisons

### RANDOMIZED-SELECT Algorithm

**Idea**: Like Quicksort but recurse only on one partition.

**Algorithm**:
1. If p = r, return A[p]
2. q ← RANDOMIZED-PARTITION(A, p, r)
3. k ← q - p + 1 (size of left partition including pivot)
4. If i = k: return A[q] (found it!)
5. If i < k: recurse on left A[p..q-1]
6. If i > k: recurse on right A[q+1..r] for (i-k)-th element

**Running Time**:
- **Worst case**: Θ(n²) - always partition around largest/smallest
- **Expected case**: **Θ(n)** - assumed i-th element always in larger partition

### SELECT in Worst-Case O(n)

**Median-of-Medians Approach**:

1. **Divide**: n elements into groups of 5 → ⌈n/5⌉ groups
2. **Find medians**: Sort each group (O(1) each), pick median
3. **Recursively**: Find median x of the ⌈n/5⌉ medians
4. **Partition**: Around median-of-medians x
5. **Recurse**: On appropriate partition

**Key Insight**: At least 3n/10 - 6 elements guaranteed ≤ x, and at least 3n/10 - 6 guaranteed ≥ x.

**Analysis**: 
- Larger partition has at most **7n/10 + 6 elements**
- Recurrence: T(n) = T(⌈n/5⌉) + T(7n/10 + 6) + O(n)
- Solution: **T(n) = O(n)**

---

## 4. Sorting in Linear Time

### Lower Bound for Comparison Sorts

**Theorem**: Any comparison sort requires **Ω(n lg n)** comparisons in worst case.

**Proof** (Decision Tree Model):
- Decision tree models all possible comparisons
- n! permutations → at least n! leaves
- Tree height h ≥ lg(n!) = Ω(n lg n)

**To beat Ω(n lg n)**: Use operations other than comparisons!

---

### Counting Sort

**Assumption**: Elements are integers in range **0 to k**.

**Idea**:
- For each element x, count elements ≤ x
- Place x directly in correct output position

**Algorithm**:
1. Count occurrences: C[i] = number of elements equal to i
2. Compute cumulative: C[i] = number of elements ≤ i
3. Place elements: Use C to determine positions

**Properties**:
- **Running time**: Θ(n + k)
- **When k = O(n)**: Running time is **Θ(n)**
- **Stable**: Elements with same value maintain relative order

**Why stability matters**: Important when keys carry additional data.

---

### Radix Sort

**Idea**: Sort d-digit numbers by processing digits from **least significant to most significant**.

**Algorithm**:
```
For i = 1 to d:
    Use stable sort on digit i
```

**Properties**:
- Must use **stable sort** for each digit (e.g., counting sort)
- Each digit has k possible values
- **Running time**: Θ(d(n + k))
- d passes through array

**Correctness**: By induction on number of passes
- After sorting on digits 1..j, elements are sorted on those digits
- Stable sort preserves this when sorting digit j+1

---

### Bucket Sort

**Assumption**: Input uniformly distributed over **[0, 1)**.

**Algorithm**:
1. Divide [0, 1) into n equal buckets
2. Distribute n input values into buckets
3. Sort each bucket (using insertion sort)
4. Concatenate buckets in order

**Element placement**: A[i] goes into bucket ⌊n·A[i]⌋

**Properties**:
- **Average-case running time**: **Θ(n)**
- Works well when input is uniformly distributed
- Sorting buckets takes O(1) expected time each

**Correctness**: 
- If A[i] ≤ A[j], then ⌊n·A[i]⌋ ≤ ⌊n·A[j]⌋
- Elements in same bucket sorted by insertion sort
- Concatenation preserves order

---

## 5. Heaps

### Heap Properties

**Definition**: Nearly complete binary tree with:

1. **Structural property**: All levels full except possibly last, filled left to right
2. **Max-heap property**: Parent(x) ≥ x for all nodes x

**Height**: h = ⌊lg n⌋ for heap with n elements

**Min-heap property** (alternative): Parent(x) ≤ x

### Array Representation

Store heap as array A with:
- **Root**: A[1]
- **Left child** of A[i]: A[2i]
- **Right child** of A[i]: A[2i + 1]
- **Parent** of A[i]: A[⌊i/2⌋]
- **Leaves**: A[⌊n/2⌋ + 1..n]

### MAX-HEAPIFY

**Preconditions**:
- Left and right subtrees of i are max-heaps
- A[i] may be smaller than children

**Algorithm**:
1. Find largest among A[i], A[left[i]], A[right[i]]
2. If largest ≠ i:
   - Exchange A[i] with largest child
   - Recursively heapify that child

**Running time**: **O(lg n)** or O(h) where h is height

---

### BUILD-MAX-HEAP

**Goal**: Convert unordered array into max-heap.

**Algorithm**:
```
For i = ⌊n/2⌋ downto 1:
    MAX-HEAPIFY(A, i, n)
```

**Why start at ⌊n/2⌋?** Elements from ⌊n/2⌋+1 to n are leaves.

**Loop Invariant**: At start of each iteration, nodes i+1, i+2, ..., n are roots of max-heaps.

**Running Time**: **O(n)** (tight bound!)
- NOT O(n lg n) as naive analysis suggests
- Proof: Most nodes are near bottom, have small height

**Analysis**:
- At most ⌈n/2^(h+1)⌉ nodes of height h
- Summation: Σ(h=0 to ⌊lg n⌋) ⌈n/2^(h+1)⌉ · O(h) = O(n)

---

### HEAPSORT

**Algorithm**:
1. BUILD-MAX-HEAP(A) - O(n)
2. For i = n downto 2:
   - Exchange A[1] with A[i] (put max at end)
   - heap-size[A] = heap-size[A] - 1
   - MAX-HEAPIFY(A, 1) - O(lg n)

**Running Time**: **O(n lg n)**

**Properties**:
- Sorts in place
- NOT stable
- Worse constants than Quicksort in practice

---

### Priority Queues

**Max-Priority Queue**: Data structure supporting:
- INSERT(S, x): Insert x into set S
- MAXIMUM(S): Return element with largest key
- EXTRACT-MAX(S): Remove and return largest element
- INCREASE-KEY(S, x, k): Increase x's key to k

**Heap Implementation**:

**HEAP-MAXIMUM(A)**: Return A[1] - **O(1)**

**HEAP-EXTRACT-MAX(A)**:
1. Save max = A[1]
2. A[1] = A[heap-size[A]]
3. heap-size[A] = heap-size[A] - 1
4. MAX-HEAPIFY(A, 1)
5. Return max

**Running time**: **O(lg n)**

---

**HEAP-INCREASE-KEY(A, i, key)**:
1. If key < A[i]: error
2. A[i] = key
3. While i > 1 and A[parent[i]] < A[i]:
   - Exchange A[i] with A[parent[i]]
   - i = parent[i]

**Running time**: **O(lg n)** - path to root

**MAX-HEAP-INSERT(A, key)**:
1. heap-size[A] = heap-size[A] + 1
2. A[heap-size[A]] = -∞
3. HEAP-INCREASE-KEY(A, heap-size[A], key)

**Running time**: **O(lg n)**

---

## 6. Red-Black Trees

### Properties

**Red-Black Tree**: Binary search tree with one extra bit per node (color: red or black).

**RBT Properties**:
1. Every node is red or black
2. **Root is black**
3. **Every leaf (NIL) is black**
4. **Red node has black children** (no two consecutive reds)
5. **All paths from node to leaves have same black-height**

**Black-height bh(x)**: Number of black nodes on path to leaf (not counting x).

**Height Bound**: h ≤ 2 lg(n + 1)

**Proof**: Subtree with bh(x) has ≥ 2^(bh(x)) - 1 internal nodes.

---

### Rotations

**Purpose**: Maintain BST property while restructuring tree.

**LEFT-ROTATE(T, x)**:
- Makes right child y the new root of subtree
- x becomes y's left child
- y's left subtree becomes x's right subtree

**RIGHT-ROTATE(T, y)**: Inverse of left-rotate

**Time**: **O(1)** - constant pointer changes

**Key**: Rotations preserve BST property and modify structure.

---

### RB-INSERT

**Basic Idea**:
1. Insert like normal BST
2. Color new node **red**
3. Fix violations moving up tree

**Violations**:
- Red root (easy: color black)
- Red node with red child (need rotation/recoloring)

**Three Cases** (z is red node with red parent):

**Case 1**: z's uncle y is **red**
- Color parent **black**, uncle **black**, grandparent **red**
- Move z to grandparent
- **Continue** loop

**Case 2**: z's uncle y is **black**, z is **right child**
- LEFT-ROTATE(T, parent)
- Move z to former parent
- **Go to Case 3**

**Case 3**: z's uncle y is **black**, z is **left child**
- Color parent **black**, grandparent **red**
- RIGHT-ROTATE(T, grandparent)
- **Done!**

**Finally**: Color root black

**Running Time**: **O(lg n)**
- While loop executes O(lg n) times (only Case 1 repeats)
- Each iteration O(1)

---

## 7. Order-Statistic Trees (OS-Trees)

### Augmentation

Red-black tree with additional field **size[x]** for each node:
- size[x] = number of internal nodes in subtree rooted at x (including x)
- **Maintained**: size[x] = size[left[x]] + size[right[x]] + 1

### OS-SELECT(x, i)

**Goal**: Return pointer to node with i-th smallest key in subtree rooted at x.

**Algorithm**:
1. r = size[left[x]] + 1 (rank of x in its subtree)
2. If i = r: return x
3. If i < r: return OS-SELECT(left[x], i)
4. If i > r: return OS-SELECT(right[x], i - r)

**Running Time**: **O(lg n)** - descends tree

### OS-RANK(T, x)

**Goal**: Return rank of x in linear order of tree T.

**Algorithm**:
1. r = size[left[x]] + 1
2. y = x
3. While y ≠ root[T]:
   - If y = right[p[y]]:
     - r = r + size[left[p[y]]] + 1
   - y = p[y]
4. Return r

**Running Time**: **O(lg n)** - traverses up tree

### Maintaining Size During Modifications

**INSERT**:
- **Phase 1** (going down): Increment size[x] for each node x on path
- **Phase 2** (going up): Rotations may occur
  - After rotation, recompute size using formula
  - Constant work per rotation

**Rotations**: 
- After LEFT-ROTATE or RIGHT-ROTATE, update affected sizes
- Use: size[x] = size[left[x]] + size[right[x]] + 1

**Total time**: Still **O(lg n)**

### Theorem on Augmenting RBTs

**Theorem**: Let f be a field augmenting a red-black tree. If f for node x can be computed using only information in x, left[x], right[x], then we can maintain f during insert/delete in **O(lg n)** time.

**Examples**:
- ✅ size[x]: Computable from left, right sizes
- ✅ height[x]: Computable from left, right heights  
- ❌ rank[x]: Inserting new min changes all n ranks

---

## 8. Interval Trees

### Purpose
Maintain dynamic set of **intervals**, support overlap queries.

### Interval Properties

**Intervals i and j overlap** iff:
- low[i] ≤ high[j] **AND** low[j] ≤ high[i]

**Intervals do not overlap** iff:
- high[i] < low[j] **OR** high[j] < low[i]

**Interval Trichotomy**: Exactly one holds:
1. i and j overlap
2. i is to the left of j (high[i] < low[j])
3. i is to the right of j (high[j] < low[i])

### Structure

**Base**: Red-black tree
- **Key**: low[int[x]] (low endpoint of interval)
- Each node stores: interval int[x]
- Inorder walk lists intervals sorted by low endpoint

**Augmentation**: 
- max[x] = maximum endpoint value in subtree rooted at x
- **Formula**: max[x] = max(high[int[x]], max[left[x]], max[right[x]])

### INTERVAL-SEARCH(T, i)

**Goal**: Find node x where int[x] overlaps interval i (or return NIL).

**Algorithm**:
```
x = root[T]
While x ≠ NIL and i does not overlap int[x]:
    If left[x] ≠ NIL and max[left[x]] ≥ low[i]:
        x = left[x]
    Else:
        x = right[x]
Return x
```

**Running Time**: **O(lg n)**

### Correctness of INTERVAL-SEARCH

**Theorem**: When search goes right, either:
- There is an overlap in right subtree, OR
- There is no overlap in either subtree

**Proof** (going right case):
- Went right because: left[x] = NIL or max[left[x]] < low[i]
- If max[left[x]] < low[i], then all intervals in left subtree have high < low[i]
- No overlap possible in left

**Similar theorem** when going left.

**Conclusion**: Safe to proceed in only one direction.

---

## Study Tips and Advice

### Understanding Over Memorization

- ✅ **Understand** how algorithms work conceptually
- ✅ Work through **examples** from class lectures
- ✅ Be able to **narrate** main steps in your own words
- ✅ Know **when** each algorithm is applicable
- ✅ Practice **loop invariants** and correctness proofs
- ❌ Do NOT just memorize algorithms

### Effective Study Strategies

1. **Work through examples**: 
   - Trace algorithms step-by-step
   - Create your own small examples
   
2. **Understand key insights**:
   - Why does randomization help Quicksort?
   - Why can we beat O(n lg n) with counting sort?
   - How do rotations maintain RBT properties?

3. **Practice analysis**:
   - Recurrences and their solutions
   - Probability calculations
   - Loop invariant proofs

4. **Connect concepts**:
   - How are heaps and priority queues related?
   - How does augmentation extend RBT capabilities?

---

## Quick Reference: Running Times

### Sorting Algorithms

| Algorithm | Time | Notes |
|-----------|------|-------|
| Randomized Quicksort | Expected O(n lg n), Worst Θ(n²) | Average case analysis |
| Counting Sort | Θ(n + k) | When k = O(n), time is Θ(n) |
| Radix Sort | Θ(d(n + k)) | d digits, k values per digit |
| Bucket Sort | Average Θ(n) | Uniform distribution |
| Heapsort | O(n lg n) | In-place |

### Selection Algorithms

| Operation | Time | Notes |
|-----------|------|-------|
| Min or Max | n-1 comparisons | Optimal |
| Simultaneous Min/Max | 3n/2 comparisons | Process in pairs |
| Randomized-Select | Expected Θ(n), Worst Θ(n²) | Average case |
| Worst-case Select | O(n) | Median-of-medians |

### Heap Operations

| Operation | Time |
|-----------|------|
| MAX-HEAPIFY | O(lg n) |
| BUILD-MAX-HEAP | O(n) |
| HEAPSORT | O(n lg n) |
| HEAP-EXTRACT-MAX | O(lg n) |
| HEAP-INCREASE-KEY | O(lg n) |
| MAX-HEAP-INSERT | O(lg n) |
| HEAP-MAXIMUM | O(1) |

### Red-Black Trees & Augmented Structures

| Operation | Time |
|-----------|------|
| Search, Insert, Delete | O(lg n) |
| RB-INSERT | O(lg n) |
| Rotations | O(1) |
| OS-SELECT | O(lg n) |
| OS-RANK | O(lg n) |
| INTERVAL-SEARCH | O(lg n) |

---

## Important Formulas and Bounds

### Probability
- E[I{A}] = Pr{A} (indicator random variable)
- E[X] = Σₓ x·Pr{X = x}

### Trees
- Complete binary tree height: ⌊lg n⌋
- RBT height: h ≤ 2 lg(n+1)
- Subtree with bh(x) has ≥ 2^(bh(x)) - 1 nodes

### Summations
- Σᵢ₌₁ⁿ i = n(n+1)/2
- Σᵢ₌₀ⁿ 2ⁱ = 2^(n+1) - 1
- lg(n!) = Θ(n lg n)

---

**Good luck on your exam!** Remember to show your work and explain your reasoning clearly.