---
title: "How to implement Permutation in python"
excerpt: "A practice to implmentation of permutation."
categories:
 - algorithms
tags:
 - enumerate
use_math: true
last_modified_at: "2020-06-02"
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
import sys
sys.path.append("/home/swyoo/algorithm/")
import random
from copy import deepcopy
from utils.verbose import logging_time
size = 3
a = list(range(size))
a
```

</div>




{:.output_data_text}

```
[0, 1, 2]
```



# Permutation

## Recursive Approach
<img src="https://media.geeksforgeeks.org/wp-content/cdn-uploads/NewPermutation.gif" width="400">

The time complexity is $O(nn!)$ <br>
Note that there are $n!$ **permutations** and it requires $O(n)$ time **to print a a permutation**.

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
@logging_time
def solution1(a):
    perms = []
    def f(s, e):
        if s == e:
            perms.append(deepcopy(a))
        for i in range(s, e + 1):
            a[i], a[s] = a[s], a[i]
            f(s + 1, e)
            a[i], a[s] = a[s], a[i]
    f(0, len(a) - 1)
    return perms
        
solution1(deepcopy(a), verbose=True)
```

</div>

{:.output_stream}

```
WorkingTime[solution1]: 0.02956 ms

```




{:.output_data_text}

```
[[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 1, 0], [2, 0, 1]]
```



## Heap's Algorithm

The time complexity of this algorithm is $O(n!)$.

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
@logging_time
def solution2(a):
    perms = []
    def f(size):
        if size == 1:
            perms.append(deepcopy(a))
            return
        
        for i in range(size):
            f(size - 1)
            if size & 1:  # odd
                a[0], a[size - 1] = a[size - 1], a[0]
            else:
                a[i], a[size - 1] = a[size - 1], a[i]
    f(len(a))
    return perms

solution2(deepcopy(a), verbose=True)
```

</div>

{:.output_stream}

```
WorkingTime[solution2]: 0.02432 ms

```




{:.output_data_text}

```
[[0, 1, 2], [1, 0, 2], [2, 0, 1], [0, 2, 1], [1, 2, 0], [2, 1, 0]]
```



<div class="prompt input_prompt">
In&nbsp;[8]:
</div>

<div class="input_area" markdown="1">

```python
size = 9
a = list(range(size))
ans1 = solution1(deepcopy(a), verbose=True)
ans2 = solution2(deepcopy(a), verbose=True)
```

</div>

{:.output_stream}

```
WorkingTime[solution1]: 2472.30506 ms
WorkingTime[solution2]: 2236.54580 ms

```

## Library

`itertools` library provides us to get permutations easily. 

<div class="prompt input_prompt">
In&nbsp;[11]:
</div>

<div class="input_area" markdown="1">

```python
from itertools import permutations

list(permutations(range(3), 2))
```

</div>




{:.output_data_text}

```
[(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
```



# Reference
[1] [Korean Minusi's blog](https://minusi.tistory.com/entry/%EC%88%9C%EC%97%B4-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-Permutation-Algorithm) <br>
[2] [Naive Permutation - geeksforgeeks](https://www.geeksforgeeks.org/write-a-c-program-to-print-all-permutations-of-a-given-string/) <br>
[3] [Heap's Algorithm - geeksforgeeks](https://www.geeksforgeeks.org/heaps-algorithm-for-generating-permutations/) <br>
[4] [Permutation with Duplicates - geeksforgeeks](https://www.geeksforgeeks.org/print-all-permutations-of-a-string-with-duplicates-allowed-in-input-string/?ref=lbp) <br>
