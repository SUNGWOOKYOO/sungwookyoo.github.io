---
title: "315.Count Smaller Numbers After Self"
excerpt: "segment Tree, merge sort are utilized. "
categories:
 - algorithms
tags:
 - datastructure
 - DivideConquer
use_math: true
last_modified_at: "2020-05-13"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/algorithms/algo.png
 overlay_filter: 0.5
---

<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
from typing import List
from treelib import Tree
from pprint import pprint
from collections import deque
import random, sys, copy
sys.path.append('/home/swyoo/algorithm/')
from utils.verbose import logging_time
```

</div>

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
INF = 1e20
n = 10000
A = [random.randint(-1e10, 1e10) for i in range(n)]  # sample
```

</div>

# 315. Count of Smaller Numbers After Self

You are given an integer array nums and you have to return a new counts array. <br>
The counts array has the property where `counts[i]` is the number of smaller elements to the right of `nums[i]`.

3 Approach exists. [reference discuss in leetcode](https://leetcode.com/problems/count-of-smaller-numbers-after-self/discuss/408322/Python-Different-Concise-Solutions)

## Naive 
Enumerate all cases.
$$
O(n^2)
$$

Naive way is too slow, TLE(Time Limited Error) occurs! when it is submitted.

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
class Solution:
    @logging_time
    def countSmaller(self, nums: List[int]) -> List[int]:
        if not nums: return []
        ans = []
        for i in range(len(nums)):
            cnt = 0
            for j in range(i + 1, len(nums)):
                if nums[i] > nums[j]:
                     cnt += 1
            ans.append(cnt)
        return ans
    
sol1 = Solution()
ans1 = sol1.countSmaller(A, verbose=True)
```

</div>

{:.output_stream}

```
WorkingTime[countSmaller]: 4177.07729 ms

```

## Segment Tree
### Idea
Count smaller numbers from `nums[i + 1:]` than `nums[i]`. <br>
Implement as follows. <br>
* This algorithm **search `nums` reversely** because we can avoid trivial things. 
* The SegmentTree copy nums to be distint in an sorted order and every nodes has `low`, `high` values in the given range. 
     * This is because we can find `cnt` by checking the `low` and `high` in $ O(logn)$.  
* **Every nodes has `cnt`**, which helps to find smaller numbers when quering. 
* **After querying time, update nodes' `cnt`** related to smaller than `nums[i]`. 
    
$$
O(nlogn)
$$

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
class Node:
    def __init__(self, low, high):
        self.low, self.high = low, high
        self.left = self.right = None
        self.cnt = 0

    def __repr__(self) -> str:
        if self.low != self.high: 
            return "[{}-{}]^{}".format(self.low, self.high, self.cnt)  
        else: 
            return "[{}]^{}".format(self.low, self.cnt)


class SegTree:
    def __init__(self, nums):
        self.nums = nums
        self.root = self.build()

    def build(self):
        def _build(s, e):
            cur = Node(low=self.nums[s], high=self.nums[e])
            if s == e:
                return cur
            mid = (s + e) // 2
            cur.left, cur.right = _build(s, mid), _build(mid + 1, e)
            return cur

        return _build(0, len(self.nums) - 1)

    def query(self, p, r):
        def _query(cur: Node):
            if r < cur.low or p > cur.high:
                return 0
            if p <= cur.low and cur.high <= r:
                return cur.cnt
            return _query(cur.left) + _query(cur.right)

        return _query(self.root)

    def update(self, x):
        """ update nodes related to x. """

        def _update(cur):
            if not cur:
                return 0
            if cur.low <= x <= cur.high:
                cur.cnt += 1
                _update(cur.left), _update(cur.right)

        _update(self.root)

    def show(self):
        s = self.root
        queue = deque([s])
        tree = Tree()
        tree.create_node(tag=str(s), identifier=s)
        while queue:
            u = queue.popleft()
            if u:
                for v in [u.left, u.right]:
                    queue.append(v)
                    tree.create_node(tag=str(v),
                                     identifier=v,
                                     parent=u)
        return str(tree.show())


class Solution:
    @logging_time
    def countSmaller(self, nums: List[int], show=False) -> List[int]:
        if not nums: return []
        nums = nums[::-1]
        st = SegTree(sorted(list(set(nums))))
        if show:
            print("BEFORE UPDATE")
            print(st.nums)
            st.show()
        ans = []
        for e in nums:
            ans.append(st.query(-INF, e - 1))
            st.update(x=e)
        if show:
            print("AFTER UPDATE")
            st.show()
        return ans[::-1]

sol2 = Solution()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
ans2 = sol2.countSmaller(A, verbose=True)
```

</div>

{:.output_stream}

```
WorkingTime[countSmaller]: 382.06029 ms

```

### Visualizaion

Please note that it is not same with general segment tree. <br>
General SegmentTree does not have `self.nums`, <br>
where they are copied and then duplicates are removed and sorted. <br>
Internal nodes of general SegmentTree are targeted values (like max value, min value, etc) <br>
within a given range of index in the node. <br>
In this problem, Internel nodes have partial range of `self.nums` and the `low` and `high` of `self.nums`. <br>

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
assert ans1 == ans2, "Error"
```

</div>

<div class="prompt input_prompt">
In&nbsp;[7]:
</div>

<div class="input_area" markdown="1">

```python
toy_example = [random.randint(1, 20) for i in range(5)]  # sample
print(toy_example)
sol2.countSmaller(toy_example, verbose=True, show=True)
```

</div>

{:.output_stream}

```
[15, 12, 20, 14, 3]
BEFORE UPDATE
[3, 12, 14, 15, 20]
[3-20]^0
├── [15-20]^0
│   ├── [15]^0
│   │   ├── None
│   │   └── None
│   └── [20]^0
│       ├── None
│       └── None
└── [3-14]^0
    ├── [14]^0
    │   ├── None
    │   └── None
    └── [3-12]^0
        ├── [12]^0
        │   ├── None
        │   └── None
        └── [3]^0
            ├── None
            └── None

AFTER UPDATE
[3-20]^5
├── [15-20]^2
│   ├── [15]^1
│   │   ├── None
│   │   └── None
│   └── [20]^1
│       ├── None
│       └── None
└── [3-14]^3
    ├── [14]^1
    │   ├── None
    │   └── None
    └── [3-12]^2
        ├── [12]^1
        │   ├── None
        │   └── None
        └── [3]^1
            ├── None
            └── None

WorkingTime[countSmaller]: 18.37850 ms

```




{:.output_data_text}

```
[3, 1, 2, 1, 0]
```



## Merge Sort 

This idea is advance version of [Counting Inversion Problem](https://sungwookyoo.github.io/algorithms/CountInversion/). <br>
Count inversion for each elemenet! <br>
So for this, we should keep track of indices when merging. <br>
When sorting, reflect cumulative summation of targerted counters. <br>
Detailed explanation in [an article of discuss in leetcoded](https://leetcode.com/problems/count-of-smaller-numbers-after-self/discuss/76607/C%2B%2B-O(nlogn)-Time-O(n)-Space-MergeSort-Solution-with-Detail-Explanation) <br>

<div class="prompt input_prompt">
In&nbsp;[8]:
</div>

<div class="input_area" markdown="1">

```python
class Solution:
    @logging_time
    def countSmaller(self, nums: List[int]) -> List[int]:
        if not nums: return []
        indices = list(range(len(nums)))
        counts = [0] * len(nums)

        def sort(s, e):
            if s == e:
                return
            mid = (s + e) // 2
            sort(s, mid)
            sort(mid + 1, e)
            # merge
            i = j = 0
            L = copy.deepcopy(nums[s: mid + 1])
            R = copy.deepcopy(nums[mid + 1: e + 1])
            L.append(-INF), R.append(-INF)
            Lidx = copy.deepcopy(indices[s: mid + 1])
            Ridx = copy.deepcopy(indices[mid + 1: e + 1])
            for k in range(s, e + 1):
                if L[i] > R[j]:
                    nums[k] = L[i]
                    indices[k] = Lidx[i]
                    if i != len(L) - 1:
                        counts[indices[k]] += (len(R) - 1 - j)
                    i += 1
                else:
                    nums[k] = R[j]
                    indices[k] = Ridx[j]
                    j += 1
        sort(0, len(nums) - 1)
        return indices, counts
    
sol3 = Solution()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[9]:
</div>

<div class="input_area" markdown="1">

```python
ans2 = sol2.countSmaller(A, verbose=True)
indices, ans3 = sol3.countSmaller(A, verbose=True)
assert ans2 == ans3, "A={}| ans2={}, ans3={}".format(A, ans2, ans3)
```

</div>

{:.output_stream}

```
WorkingTime[countSmaller]: 269.34028 ms
WorkingTime[countSmaller]: 308.29358 ms

```

<div class="prompt input_prompt">
In&nbsp;[10]:
</div>

<div class="input_area" markdown="1">

```python
print(toy_example)
ans2 = sol2.countSmaller(toy_example, verbose=True)
indices, ans3 = sol3.countSmaller(toy_example, verbose=True)
print("after sorted:{}, indices:{}".format(toy_example, indices))
assert ans2 == ans3, "toy_example={}| ans2={}, ans3={}".format(toy_example, ans2, ans3)
```

</div>

{:.output_stream}

```
[15, 12, 20, 14, 3]
WorkingTime[countSmaller]: 0.04125 ms
WorkingTime[countSmaller]: 0.05770 ms
after sorted:[20, 15, 14, 12, 3], indices:[2, 0, 3, 1, 4]

```
