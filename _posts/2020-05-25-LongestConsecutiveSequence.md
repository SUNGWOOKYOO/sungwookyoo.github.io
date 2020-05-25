---
title: "128.Longest Consecutive Sequences"
excerpt: "practice of disjoint set"
categories:
 - algorithms
tags:
 - datastructure
 - union find
use_math: true
last_modified_at: "2020-05-25"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/teaser.jpg
 overlay_filter: 0.5
---

<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
from typing import List
import sys, random
sys.path.append("/home/swyoo/algorithm/")
from utils.verbose import logging_time, visualize_ds
```

</div>

# 128. Longest Consecutive Sequence

## Disjoint Set(or Union Find)
### With Sorting
$$
O(nlogn)
$$

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
# Disjoint Set data structure
par, rnk = {}, {}
cnt = {}
def find(x):
    if x not in par:
        par[x] = x
        rnk[x] = 0
        cnt[x] = 1
        return par[x]
    if x != par[x]:
        par[x] = find(par[x])
    return par[x]

def union(x, y):
    x, y = find(x), find(y)
    if x == y: return
    if rnk[x] > rnk[y]:
        x, y = y, x
    par[x] = y
    cnt[y] += cnt[x]
    if rnk[x] == rnk[y]:
        rnk[y] += 1
```

</div>

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
class Solution1:
    @logging_time
    def longestConsecutive(self, nums: List[int], show=False) -> int:
        if not nums: return 0
        nums = sorted(nums)
        for i in range(1, len(nums)):
            if nums[i] - nums[i - 1] == 1:
                union(nums[i - 1], nums[i])
        if show: visualize_ds(par)
        return max(cnt.values()) if cnt else 1

sol1 = Solution1()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
A = [100,4,200,1,3,2]
print(sol1.longestConsecutive(A, show=True, verbose=True))
```

</div>


![png](/assets/images/LongestConsecutiveSequence_files/LongestConsecutiveSequence_4_0.png)


{:.output_stream}

```
WorkingTime[longestConsecutive]: 173.99216 ms
4

```

### Without Sorting
If the algorithm use some techniques as follows, it can be improved. 
* Conversion of `nums` from list to set: checking if exist takes $O(1)$. 
* Both 'path compression' ans 'union by rank' 

The time complexity of the whole process becomes as follows. <br>
$$
O(n\alpha)
$$

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
class Solution2:
    @logging_time
    def longestConsecutive(self, nums: List[int], show=False) -> int:
        if not nums: return 0
        nums = set(nums)
        used = set()
        for e in nums:
            if e not in used and e - 1 in nums:
                union(e - 1, e)
                used.add(e)
        if show: visualize_ds(par)
        return max(cnt.values()) if cnt else 1
sol2 = Solution2()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
par, rnk, cnt = {}, {}, {}
print(sol2.longestConsecutive(A, show=True, verbose=True))
```

</div>


![png](/assets/images/LongestConsecutiveSequence_files/LongestConsecutiveSequence_7_0.png)


{:.output_stream}

```
WorkingTime[longestConsecutive]: 162.27818 ms
4

```

<div class="prompt input_prompt">
In&nbsp;[7]:
</div>

<div class="input_area" markdown="1">

```python
size = 10000000
A = [random.randint(-size*10, size*10) for i in range(size)]
```

</div>

<div class="prompt input_prompt">
In&nbsp;[8]:
</div>

<div class="input_area" markdown="1">

```python
par, rnk, cnt = {}, {}, {}
ans1 = sol1.longestConsecutive(A, verbose=True) 
par, rnk, cnt = {}, {}, {}
ans2 = sol2.longestConsecutive(A, verbose=True)
```

</div>

{:.output_stream}

```
WorkingTime[longestConsecutive]: 6538.45930 ms
WorkingTime[longestConsecutive]: 3892.41171 ms

```

## Without Disjoint Set

I imported a solution from a document of the [discuss](https://leetcode.com/problems/longest-consecutive-sequence/discuss/41057/Simple-O(n)-with-Explanation-Just-walk-each-streak)

<div class="prompt input_prompt">
In&nbsp;[9]:
</div>

<div class="input_area" markdown="1">

```python
class Solution3:
    @logging_time
    def longestConsecutive(self, nums):
        nums = set(nums)
        best = 0
        for x in nums:
            if x - 1 not in nums:
                y = x + 1
                while y in nums:
                    y += 1
                best = max(best, y - x)
        return best
sol3 = Solution3()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[10]:
</div>

<div class="input_area" markdown="1">

```python
ans3 = sol3.longestConsecutive(A, verbose=True)
```

</div>

{:.output_stream}

```
WorkingTime[longestConsecutive]: 4654.11186 ms

```

<div class="prompt input_prompt">
In&nbsp;[11]:
</div>

<div class="input_area" markdown="1">

```python
ans1, ans2, ans3
```

</div>




{:.output_data_text}

```
(6, 6, 6)
```


