---
title: "Find Top K sth"
excerpt: "sorting and heap usage practice"
categories:
 - algorithms
tags:
 - heap
 - enumerate
 - DivideConquer
use_math: true
last_modified_at: "2020-05-27"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/teaser.jpg
 overlay_filter: 0.5
 caption: #
 actions:
  - label: "#"
    url: "#"
---

<div class="prompt input_prompt">
In&nbsp;[19]:
</div>

<div class="input_area" markdown="1">

```python
from collections import Counter
from typing import List
import heapq, random
import numpy as np
from itertools import chai
```

</div>

# 692.Top K Frequent Words

## Hashmap + Sorting

Given $n$ words,
1. Build Hashmap: $O(n)$
2. Sorting: $O(nlogn)$

Therefore, it takes $O(nlogn)$.

<div class="prompt input_prompt">
In&nbsp;[7]:
</div>

<div class="input_area" markdown="1">

```python
class Solution1:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        ct = Counter(words)
        return sorted(ct.keys(), key=lambda x: (-ct[x], x))[:k]
    
sol1 = Solution1()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
words = ["i", "love", "leetcode", "i", "love", "coding"]
k = 3
print(sol.topKFrequent(words, k))
```

</div>

{:.output_stream}

```
['i', 'love', 'coding']

```

## Hashmap + Heap

1. Build Hashmap takes $O(n)$
2. Build Heap by counts: $O(n)$
3. pop K words from the Heap: $O(klogn)$, where $k \le n$

Therefore, $O(n + klogn)$

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
class Solution2:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        ct = Counter(words)
        heap = [(-v, k) for k, v in ct.items()]
        heapq.heapify(heap)
        return [heapq.heappop(heap)[1] for _ in range(k)]
    
sol2 = Solution2()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[8]:
</div>

<div class="input_area" markdown="1">

```python
words = ["i", "love", "leetcode", "i", "love", "coding"]
k = 3
print(sol.topKFrequent(words, k))
```

</div>

{:.output_stream}

```
['i', 'love', 'coding']

```

# 451. Sort Characters By Frequency

<div class="prompt input_prompt">
In&nbsp;[9]:
</div>

<div class="input_area" markdown="1">

```python
class Solution:
    def frequencySort(self, s: str) -> str:
        cnt = Counter(s)
        reorder = sorted(cnt.keys(), key=lambda x: -cnt[x])
        ans = ''
        for c in reorder:
            ans += c * cnt[c]
        return ans

sol = Solution()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[10]:
</div>

<div class="input_area" markdown="1">

```python
s = "tree"
print(sol.frequencySort(s))
```

</div>

{:.output_stream}

```
eetr

```

<div class="prompt input_prompt">
In&nbsp;[11]:
</div>

<div class="input_area" markdown="1">

```python
class Solution:
    def frequencySort(self, s: str) -> str:
        cnt = Counter(s)
        heap = [(-v, k) for k, v in cnt.items()]
        heapq.heapify(heap)
        ans = ''
        while heap:
            c = heapq.heappop(heap)[1]
            ans += c * cnt[c]
        return ans

sol = Solution()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[13]:
</div>

<div class="input_area" markdown="1">

```python
s = "tree"
print(sol.frequencySort(s))
```

</div>

{:.output_stream}

```
eert

```

# 347. Top K Frequent Elements

The algorithm which use heap datas tructure takes $O(n + klogn)$

<div class="prompt input_prompt">
In&nbsp;[14]:
</div>

<div class="input_area" markdown="1">

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        cnt = Counter(nums)
        heap = [(-v, k) for k, v in cnt.items()]
        heapq.heapify(heap)
        return [heapq.heappop(heap)[1] for _ in range(k)]

sol = Solution()
nums = [1,1,1,2,2,3]
k = 2
print(sol.topKFrequent(nums, k))
```

</div>

{:.output_stream}

```
[1, 2]

```

# 378. Kth Smallest Element in a Sorted Matrix

Given $n \times n$ `matrix`, where the each rows and columns of the matrix is sorted in ascending order. <br>
condition: $1 \le k \le n^2$

## Unpack and Sorting

Ignored sorted conditions. <br>
1. unpack: $O(n^2)$
2. sorting: $O(n^2logn^2)$ 

Therefore, $O(n^2logn^2)$

<div class="prompt input_prompt">
In&nbsp;[44]:
</div>

<div class="input_area" markdown="1">

```python
class Solution1:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        unpacked = list(chain.from_iterable(matrix))
        return sorted(unpacked)[k - 1]
sol1 = Solution1()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[45]:
</div>

<div class="input_area" markdown="1">

```python
matrix = \
[[ 1,  5,  9],
 [10, 11, 13],
 [12, 13, 15]]
k = 8
print(sol1.kthSmallest(matrix, k))
```

</div>

{:.output_stream}

```
13

```

## Unpack and Selection 

1. unpack: $O(n^2)$
2. selection(average time): $O(n^2)$ 

Therefore, $O(n^2)$

<div class="prompt input_prompt">
In&nbsp;[46]:
</div>

<div class="input_area" markdown="1">

```python
class Solution2:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        a = list(chain.from_iterable(matrix))
        def select(p, r, k):
            if p == r: return a[p]
            i = random.randint(p, r)
            a[i], a[r] = a[r], a[i]
            i = p - 1
            for j in range(p, r):
                if a[j] < a[r]:
                    i += 1
                    a[i], a[j] = a[j], a[i]
            a[i + 1], a[r] = a[r], a[i + 1]
            q = i + 1
            if q - p + 1 >= k:
                return select(p, q, k)
            else:
                return select(q + 1, r, k - (q - p + 1))
        return select(0, len(a) - 1, k)

sol2 = Solution2()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[47]:
</div>

<div class="input_area" markdown="1">

```python
print(sol2.kthSmallest(matrix, k))
```

</div>

{:.output_stream}

```
13

```

## Use Heap, indexing of each node

1. build heap: $O(n^2)$
2. pop k elements: $O(klogn)$

Therefore, $O(klogn)$, but worst case $O(n^2logn)$

For helping understanding of this algorithm, I cited an article's paragraph and exmaple as follows.
> Since the matrix is sorted, we do not need to put all the elements in heap at one time. <br>
We can simply pop and put for k times. <br>
By observation, if we look at the matrix diagonally, <br>
we can tell that if we do not pop matrix[i][j], <br>
we do not need to put on matrix[i][j + 1] and matrix[i + 1][j] since they are bigger. <br>
e.g., given the matrix below: <br>
```
1 2 4 
3 5 7 
6 8 9
```
>We put 1 first, then pop 1 and put 2 and 3, then pop 2 and put 4 and 5, then pop 3 and put 6...

I cited from [this article](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/discuss/85219/Python-solution-O(klogk)-similar-to-problem-373)

<div class="prompt input_prompt">
In&nbsp;[52]:
</div>

<div class="input_area" markdown="1">

```python
class Solution3:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        heap = [(row[0], row, 1) for row in matrix]
        heapq.heapify(heap)

        # Since we want to find kth, we pop the first k elements
        for _ in range(k - 1):
            _, r, i = heap[0]
            if i < len(r):
                heapq.heapreplace(heap, (r[i], r, i + 1))
            else:
                heapq.heappop(heap)
        return heap[0][0]

sol3 = Solution3()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[53]:
</div>

<div class="input_area" markdown="1">

```python
print(sol3.kthSmallest(matrix, k))
```

</div>

{:.output_stream}

```
13

```
