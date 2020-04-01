---
title: "How to implement SegmentTree and Practice using python"
excerpt: "Let's learn about what is segment tree and how to implement."
categories:
 - algorithms
tags:
 - datastructure
use_math: true
last_modified_at: "2020-04-01"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/algorithms/algo.png
 overlay_filter: 0.5
---

[geeksforgeeks](https://www.geeksforgeeks.org/segment-tree-set-1-sum-of-given-range/) <br>
[baekjoon](https://www.acmicpc.net/blog/view/9) <br>
[codeforce](https://codeforces.com/blog/entry/18051) <br>

<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
import random, sys
from binarytree import build
from math import ceil, log2
import numpy as np
sys.path.append("/home/swyoo/algorithm")
from utils.verbose import logging_time, printProgressBar
```

</div>

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
# toy example
a = [1, 3, 5, 7, 9, 11]
print(a)
plot = lambda a: print(build(a))
```

</div>

{:.output_stream}

```
[1, 3, 5, 7, 9, 11]

```

# Segment Tree

* full binary tree 이다.
* internal 노드는 주어진 array의 index에따라 seqment 합을 가짐.
* leaves들이 주어진 array의 element값들.

## Build SegmentTree

segment tree를 build 하는데 걸리는 시간:
$$
T(n) = 2T(n/2) + O(1) = O(n)
$$

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
def buildseg(a):
    x = ceil(log2(len(a)))
    size = 2 * (2 ** x) - 1
    st = [0] * size # segment tree, size is num of nodes.
    def _build(s, e, i):
        """
        :[s, e]: segment indices
        :i: node index
        """
        if s == e:
            st[i] = a[s]
            return a[s]
        mid = (s + e) // 2
        st[i] = _build(s, mid, 2*i + 1) + _build(mid + 1, e, 2*i + 2)
        return st[i]
    _build(0, len(a) - 1, i=0)
    return st
```

</div>

<!-- ![](https://media.geeksforgeeks.org/wp-content/cdn-uploads/segment-tree1.png){:width="300"} -->
<img src=https://media.geeksforgeeks.org/wp-content/cdn-uploads/segment-tree1.png width="300">

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
print(a)
st = buildseg(a)
plot(st)
```

</div>

{:.output_stream}

```
[1, 3, 5, 7, 9, 11]

        ______36_______
       /               \
    __9__            ___27__
   /     \          /       \
  4       5        16        11
 / \     / \      /  \      /  \
1   3   0   0    7    9    0    0


```

## Get Sum using SegmentTree
Segment Tree를 한번 만들어 놓으면 subrange sum을 구하는데 걸리는 시간 다음과 같다.
$$
T(n) = T(n/2) + O(1) = O(logn)
$$

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
@logging_time
def query(st:list, s, e, n):
    """ st is segment tree implemented by list. 
    :[s, e]: query indices, which is fixed. 
    return sum[arr[s: e + 1]] in O(logn). """
    assert 0 <= s <= e < n, "invalid"
    def _sum(p, r, i):
        """
        :[p, r]: start and last searching-indices of the segment.
        :i: current node index 
        """
        if p > e or r < s: return 0 # outside the given range.
        if s <= p and r <= e: return st[i] # segment of st[i] is a part of given range.
        # partial contained.
        mid = (p + r) // 2
        return _sum(p, mid, 2*i + 1) + _sum(mid + 1, r, 2*i + 2)
        
    return _sum(0, n - 1, i=0)

@logging_time
def func(a, s, e):
    return sum(a[s:e + 1])
```

</div>

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
# case study 
nrange = 100
ratio = 0.4
n = 10
a = [random.randint(int(- ratio * nrange), int((1 - ratio) * nrange)) for _ in range(n)]
print(a)
st = buildseg(a)
plot(st)
s, e = sorted(np.random.randint(0, n - 1, size=2))
ans = query(st, s, e, len(a), verbose=True)
gt = func(a, s, e, verbose=True)
```

</div>

{:.output_stream}

```
[31, 19, -38, 8, 48, 50, 16, 5, -19, 54]

                      _________________174___________________
                     /                                       \
           _________68______                          _______106________
          /                 \                        /                  \
     ____12___             __56__               ____71__              ___35__
    /         \           /      \             /        \            /       \
  _50         -38        8        48         _66         5         -19        54
 /   \       /   \      / \      /  \       /   \       / \       /   \      /  \
31    19    0     0    0   0    0    0     50    16    0   0     0     0    0    0

WorkingTime[query]: 0.00930 ms
WorkingTime[func]: 0.00310 ms

```

<div class="prompt input_prompt">
In&nbsp;[7]:
</div>

<div class="input_area" markdown="1">

```python
# test
nrange = 100
ratio = 0.4
n = 1000000
a = [random.randint(int(- ratio * nrange), int((1 - ratio) * nrange)) for _ in range(n)]
st = buildseg(a)
s, e = sorted(np.random.randint(0, n - 1, size=2))
ans = query(st, s, e, len(a), verbose=True)
gt = func(a, s, e, verbose=True)
print("sum[{}..{}]={}, gt={}".format(s, e, ans, gt))
assert ans == gt
```

</div>

{:.output_stream}

```
WorkingTime[query]: 0.04601 ms
WorkingTime[func]: 0.53072 ms
sum[205714..247716]=422381, gt=422381

```

## Update 

recall this figure below <br>
<img src=https://media.geeksforgeeks.org/wp-content/cdn-uploads/segment-tree1.png width="300">

$$
T(n) = T(n/2) + O(1) = O(logn)
$$

I refered [this article](https://cp-algorithms.com/data_structures/segment_tree.html) because the code is more intuitive.

<div class="prompt input_prompt">
In&nbsp;[8]:
</div>

<div class="input_area" markdown="1">

```python
def update(st, idx, x, n):
    """ update `a[idx]` into `x`. 
    :idx: updating index, which is fixed. """
    assert 0 <= idx < n, "invalid"
    a[idx] = x
    def _update(p, r, i):
        """
        :[p, r]: start and last searching-indices of the segment.
        :i: current node index 
        """
        if p == r:
            st[i] = x
            return
        mid = (p + r) // 2
        if idx <= mid: _update(p, mid, 2*i + 1)
        else: _update(mid + 1, r, 2*i + 2)
        # update internel value before finish.
        st[i] = st[2*i + 1] + st[2*i + 2]
    _update(0, n - 1, i=0)
```

</div>

<div class="prompt input_prompt">
In&nbsp;[9]:
</div>

<div class="input_area" markdown="1">

```python
# case study 
nrange = 100
ratio = 0.4
n = 10
a = [random.randint(int(- ratio * nrange), int((1 - ratio) * nrange)) for _ in range(n)]
print(a)
st = buildseg(a)
plot(st)
s, e = sorted(np.random.randint(0, n - 1, size=2))
ans = query(st, s, e, len(a), verbose=True)
gt = func(a, s, e, verbose=True)
print("sum[{}..{}]={}, gt={}".format(s, e, ans, gt))

idx = random.randint(0, n-1)
new = random.randint(int(- ratio * nrange), int((1 - ratio) * nrange))
print(), print("convert a[{}]={} into {}".format(idx, a[idx], new))
update(st, idx=idx, x=new, n=len(a))
print(a)
plot(st)
```

</div>

{:.output_stream}

```
[-38, 46, -12, 45, 18, 59, 56, 22, 47, 5]

                      __________________248______________________
                     /                                           \
           _________59_______                            ________189_______
          /                  \                          /                  \
      ___-4___             ___63__                ____137__              ___52__
     /        \           /       \              /         \            /       \
   _8         -12        45        18         _115          22         47        5
  /  \       /   \      /  \      /  \       /    \        /  \       /  \      / \
-38   46    0     0    0    0    0    0     59     56     0    0     0    0    0   0

WorkingTime[query]: 0.01121 ms
WorkingTime[func]: 0.00381 ms
sum[1..7]=234, gt=234

convert a[4]=18 into -1
[-38, 46, -12, 45, -1, 59, 56, 22, 47, 5]

                      __________________229______________________
                     /                                           \
           _________40_______                            ________189_______
          /                  \                          /                  \
      ___-4___             ___44__                ____137__              ___52__
     /        \           /       \              /         \            /       \
   _8         -12        45        -1         _115          22         47        5
  /  \       /   \      /  \      /  \       /    \        /  \       /  \      / \
-38   46    0     0    0    0    0    0     59     56     0    0     0    0    0   0


```

<div class="prompt input_prompt">
In&nbsp;[10]:
</div>

<div class="input_area" markdown="1">

```python
nums = [1,3,5]
st = buildseg(nums)
plot(st)
query(st, 2, 2, len(nums), verbose=True)
```

</div>

{:.output_stream}

```

    __9__
   /     \
  4       5
 / \     / \
1   3   0   0

WorkingTime[query]: 0.00310 ms

```




{:.output_data_text}

```
5
```



## Practice 

### Practice 1

[leetcode](https://leetcode.com/problems/range-sum-query-mutable/submissions/)

<div class="prompt input_prompt">
In&nbsp;[11]:
</div>

<div class="input_area" markdown="1">

```python
from math import ceil
from math import log

class NumArray(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.nums = nums
        if len(nums) == 0: return 
        x = int(ceil(log(len(nums)) / log(2)))
        self.st = [0] * (2 * (2 ** x) - 1)

        def build(s, e, i=0):
            if s == e:
                self.st[i] = self.nums[s]
                return self.st[i]
            mid = (s + e) // 2
            self.st[i] = build(s, mid, 2 * i + 1) + build(mid + 1, e, 2 * i + 2)
            return self.st[i]

        build(0, len(self.nums) - 1)

    def update(self, i, val):
        """
        :type i: int
        :type val: int
        :rtype: None
        """
        self.nums[i] = val
        assert 0 <= i < len(self.nums), "invalid"

        def f(p, r, idx=0):
            """
            :[p, r]: segment indices.
            :idx: node index. """
            if p == r:
                self.st[idx] = val
                return
            mid = (p + r) // 2
            left, right = 2 * idx + 1, 2 * idx + 2
            if i <= mid:
                f(p, mid, left)
            else:
                f(mid + 1, r, right)
            self.st[idx] = self.st[left] + self.st[right]

        f(0, len(self.nums) - 1)

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """

        def f(p, r, idx=0):
            if r < i or j < p: return 0
            if i <= p and r <= j: return self.st[idx]
            mid = (p + r) // 2
            left, right = 2 * idx + 1, 2 * idx + 2
            return f(p, mid, left) + f(mid + 1, r, right)

        return f(0, len(self.nums) - 1)


# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# obj.update(i,val)
# param_2 = obj.sumRange(i,j)
```

</div>

<div class="prompt input_prompt">
In&nbsp;[12]:
</div>

<div class="input_area" markdown="1">

```python
# Your NumArray object will be instantiated and called as such:
nums = [1, 3, 5]
obj = NumArray(nums)
plot(obj.st)
print(obj.sumRange(0, 2))
obj.update(1, 2)
plot(obj.st)
print(obj.sumRange(0, 2))
```

</div>

{:.output_stream}

```

    __9__
   /     \
  4       5
 / \     / \
1   3   0   0

9

    __8__
   /     \
  3       5
 / \     / \
1   2   0   0

8

```

### Practice 2

[2019 kakao internship](https://programmers.co.kr/learn/courses/30/lessons/64062) <br>
이 문제는 징검다리를 건널 수 있는 최대 사람 수를 구하는 것이 목표이다. <br>
각 사람이 점프할 수 있는 최대 길이는 `k`. <br>
한 사람이 지나갈때마다 stone 값이 -1 씩 감소하다가 연속된 `0`이 `k` 번 나올때의 사람 수를 구해야한다.  <br>

#### Key Idea 
잘 생각해보면, `stones`에서 `k`짜리 윈도우안의 <br>
`max(stones[i: i + k + 1])` 값이 그 윈도우 안에서 지나갈 수 있는 사람의 최대 수를 의미한다. 
`stones`에서 `k` 슬라이딩 윈도우를 끝까지 지나가며 구한 max 값들 중 min 값이 지나갈 수있는 사람의 최대 수이다.
$$
ans = \underset{0 \le i \le n - k}{min}(max(stones[i:i+k]))
$$

$max(stones[i: i+k])$를 구하는 방법은 naive 한 방식으로 구했을때, 걸리는 시간은 다음과 같다. 
$$
O(nk) = O(n^2) \because k \le n
$$


<div class="prompt input_prompt">
In&nbsp;[13]:
</div>

<div class="input_area" markdown="1">

```python
def solution(stones, k):
    ans = 1e13
    for i in range(len(stones) - k + 1):
        ans = min(ans, max(stones[i: i + k]))
    return ans

k = 3
stones = [2, 4, 5, 3, 2, 1, 4, 2, 5, 1]
print("ans: {}".format(solution(stones, k)))
```

</div>

{:.output_stream}

```
ans: 3

```

하지만 이 알고리즘은 효율성에서 통과를 하지 못한다. 

#### Efficient Algorithm
$max(stones[i: i+k])$를 효율적으로 구하는 방법은 여러 방법이 있겠지만, <br>
이 문제에서 효율성을 통과하기 위해서는 **segment tree**를 사용할 수 있다.  <br>
segment tree를 사용하여 일정 연속된 구간에 대한 min값을 구하도록 할 수 있다. <br>
따라서, 걸리는 시간은 다음과 같이 개선이 된다.
$$
O(nlogn)
$$

<div class="prompt input_prompt">
In&nbsp;[14]:
</div>

<div class="input_area" markdown="1">

```python
from math import ceil, log
from binarytree import build
plot = lambda t: print(build(t))

def solution(stones, k):
    x = 2 ** ceil(log(len(stones)) / log(2))
    st = [0] * (2 * x - 1)

    def _build(s, e, i=0):
        """ build segment tree, O(n). """
        if s == e:
            st[i] = stones[s]
            return st[i]
        mid = (s + e) // 2
        st[i] = max(_build(s, mid, (i << 1) + 1), _build(mid + 1, e, (i << 1) + 2))
        return st[i]

    _build(0, len(stones) - 1)
    plot(st)

    def query(s, e):
        """ find max(stones[s: e + 1]), O(logn)."""
        def _max(p, r, i=0):
            if r < s or e < p: return -1e10
            if s <= p and r <= e: return st[i]
            mid = (p + r) // 2
            return max(_max(p, mid, (i << 1) + 1),
                       _max(mid + 1, r, (i << 1) + 2))
        return _max(0, len(stones) - 1)
    
    ans = 1e13
    for i in range(len(stones) - k + 1):
        ans = min(ans, query(i, i + k - 1))
    return ans
```

</div>

<div class="prompt input_prompt">
In&nbsp;[15]:
</div>

<div class="input_area" markdown="1">

```python
k = 3
stones = [2, 4, 5, 3, 2, 1, 4, 2, 5, 1]
print("ans: {}".format(solution(stones, k)))
```

</div>

{:.output_stream}

```

                ______________5______________
               /                             \
        ______5______                   ______5______
       /             \                 /             \
    __5__           __3__           __4__           __5__
   /     \         /     \         /     \         /     \
  4       5       3       2       4       2       5       1
 / \     / \     / \     / \     / \     / \     / \     / \
2   4   0   0   0   0   0   0   1   4   0   0   0   0   0   0

ans: 3

```

# Report 

segment tree는 연속된 subrange에 대한 sum, max, min등을 log 시간으로 구할 수 있기 때문에 유용하게 사용될 수 있다. <br>
recursive 한 방식으로 구현하면, max recursion을 초과하여 runtime error가 발생 할 수 있으므로, <br>
[geeksforgeeks](https://www.geeksforgeeks.org/iterative-segment-tree-range-minimum-query/) 에서 보고 나중에 연습해보자.
