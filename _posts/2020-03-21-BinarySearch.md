---
title: "Binary Search using python"
excerpt: "implement binary search recursive and iterative."
categories:
 - algorithms
tags:
 - DivideConquer
use_math: true
last_modified_at: "2020-03-21"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/algorithm/algo.png
 overlay_filter: 0.5
 caption: algorithm
---

<div class="prompt input_prompt">
In&nbsp;[16]:
</div>

<div class="input_area" markdown="1">

```python
import sys, os, random
import numpy as np
sys.path.append("/home/swyoo/algorithm")
from utils.verbose import logging_time
```

</div>

## Recursive 
recursive version은 `x` 가 array `a`안에 존재하지 않는경우 
`s >= e` 에 들어가서 `-1` 을 return 한다. 

not exist case를 recursive로 다루는 문제 연습: [Climbing the Leaderboard](https://sungwookyoo.github.io/algorithms/Leaderboard/)

<div class="prompt input_prompt">
In&nbsp;[89]:
</div>

<div class="input_area" markdown="1">

```python
def bs(a, s, e, x):
    if s >= e:
        return s if a[s] == x else -1
    mid = (s + e) // 2
    if a[mid] < x: return bs(a, mid + 1, e, x)
    elif a[mid] == x: return mid
    else: return bs(a, s, mid -1, x)
```

</div>

<div class="prompt input_prompt">
In&nbsp;[399]:
</div>

<div class="input_area" markdown="1">

```python
a = sorted(np.random.randint(0, 500, size=15))
a
```

</div>




{:.output_data_text}

```
[95, 96, 206, 232, 249, 315, 316, 338, 399, 434, 460, 464, 465, 473, 480]
```



<div class="prompt input_prompt">
In&nbsp;[400]:
</div>

<div class="input_area" markdown="1">

```python
idx = random.randint(0, len(a) - 1)
print("find {}".format(a[idx]))
assert bs(a, 0, len(a) - 1, a[idx]) == idx
print("ans:", idx)
```

</div>

{:.output_stream}

```
find 399
ans: 8

```

<div class="prompt input_prompt">
In&nbsp;[401]:
</div>

<div class="input_area" markdown="1">

```python
bs(a, 0, len(a) - 1, random.randint(0, 500))
```

</div>




{:.output_data_text}

```
-1
```



# Iterative 

iterative version에서는 `lowerbound, upperbound`를 따로 저장해 놓을 수 있어서 <br>
recursive version보다 not exist case를 다루기 수월하다. 

<div class="prompt input_prompt">
In&nbsp;[402]:
</div>

<div class="input_area" markdown="1">

```python
def bs2(a, s, e, x):
    lowerbound = s
    upperbound = e
    while s <= e:
        mid = (s + e) // 2 
        if a[mid] == x: return mid
        if a[mid] < x: s = mid + 1
        else: e = mid - 1
    # if not present, s = e + 1
    if min(s, mid, e) < lowerbound: print("lower than a[{}]={}".format(lowerbound, a[lowerbound]))
    elif max(s, mid, e) > upperbound: print("higher than a[{}]={}".format(upperbound, a[upperbound]))
    else: print("Not exist, {} is between {} ~ {}".format(x, a[min(s, mid, e)], a[max(s, mid, e)]))
    return -1
```

</div>

<div class="prompt input_prompt">
In&nbsp;[403]:
</div>

<div class="input_area" markdown="1">

```python
print(a)
k = random.randint(0, 500)
print("pick random number {}".format(k))
bs2(a, 0, len(a) - 1, k)
```

</div>

{:.output_stream}

```
[95, 96, 206, 232, 249, 315, 316, 338, 399, 434, 460, 464, 465, 473, 480]
pick random number 40
lower than a[0]=95

```




{:.output_data_text}

```
-1
```


