---
title: "Pumping Oil using python"
excerpt: ""
categories:
 - algorithms
tags:
 - DP
use_math: true
last_modified_at: "2020-03-18"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/algorithms/algo.png
 overlay_filter: 0.5
 caption: algorithm
---

# Pumping Oil

(Divide-and-conquer) The picture below shows an oil field with $m$ by $n$ cells. Assume that $n$ is a power of $2$ (i.e., $n = 2k$). The number in each cell is the amount of oil in the cell. We want to build two towers and drills to pump out oil. The input is given as an array `A[1...m][1...n]`. We can pump out oil from the cells the drill goes through and cells between the end of two drills. The drills must be the same depth. The cost is the number of cells from which we pump oil out. <span style="color:red">**The benefit** is the amount of the oil pumped out minus the cost</span>. For example, the benefit in the picture is `(0 + 4 + 4 + 0 + 5 + 0 + 4) − 7 = 10`

![](assets/images/algorithms/pumpingoil.PNG){:width="300"}

## Divide and Conquer

<span style="color:red"> Constraint: </span> Assume that `e - s` is power of `2` so, `e - s` is always `odd`. 

### Key idea
* Find left, right optimal values using recursive call
* Find cross optimal, it takes $O(rn)$.
* Find max(left, right, cross) to determine the optimal value of `A[:r+1, s: e+1]`

For each row $r$
$$
\begin{align}
T'(r, n) 
&= 2T'(r, n/2) + O(rn) \\
&= O(rnlogn)
\end{align}
$$

So, $T(m,n) = \sum_{r=1}^m {T'(r, n)} = O(m^2nlogn) $

<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
import numpy as np
A = np.array([[0,3,4,0],[4,3,0,3],[4,0,5,3]])
print(A, 'of shape =',A.shape)

# Assume that e - s is power of 2 so, e - s is always odd. 
def oil(A, r, s, e):
    print('oil(r={}, s={}, e={})'.format(r, s, e))
    # base case
    if e - s == 1:
        neighbor = A[:(r+1), s].sum() + A[:(r+1), e].sum() - 2*(r+1)
        print('basecase! return {}'.format(neighbor))
        return (neighbor)
    
    mid = (s + e)// 2 
    left = oil(A, r, s, mid)
    right = oil(A, r, mid + 1, e)
    print('left={}, right={}'.format(left, right))
    
    # cross case computed in O(rn) time
    G = A[:(r+1), s: e+1].sum(axis=0)
    max_left, max_right = -1e+8, -1e+8
    for i in range(mid , s-1, -1):
        cal = G[i] - (mid - i) - (r+1)
        if cal > max_left:
            max_left = cal
            print('@ i={}, updated to {}'.format(i, cal))
    for j in range(mid+1, e+1):
        cal = G[j] - (j - mid - 1) - (r+1)
        if cal > max_right:
            print('@ j={}, updated to {}'.format(j, cal))
            max_right = cal
    print('max_left={}, max_right={}'.format(max_left, max_right))
    return max(left, right, max_left + max_right)

def pumping_oil(A, r, m, n):
    opt = 0
    for k in range(r+1):
        print('============= {} th row ============================'.format(k))
        this = oil(A, k, m, n)
        if this > opt:
            print('update opt value to {}'.format(this))
            opt = this
        print('===================================================='.format(k))
    return opt
```

</div>

{:.output_stream}

```
[[0 3 4 0]
 [4 3 0 3]
 [4 0 5 3]] of shape = (3, 4)

```

<div class="prompt input_prompt">
In&nbsp;[56]:
</div>

<div class="input_area" markdown="1">

```python
pumping_oil(A, 2, 0, 3)
```

</div>

{:.output_stream}

```
============= 0 th row ============================
oil(r=0, s=0, e=3)
oil(r=0, s=0, e=1)
basecase! return 1
oil(r=0, s=2, e=3)
basecase! return 2
left=1, right=2
@ i=1, updated to 2
@ j=2, updated to 3
max_left=2, max_right=3
update opt value to 5
====================================================
============= 1 th row ============================
oil(r=1, s=0, e=3)
oil(r=1, s=0, e=1)
basecase! return 6
oil(r=1, s=2, e=3)
basecase! return 3
left=6, right=3
@ i=1, updated to 4
@ j=2, updated to 2
max_left=4, max_right=2
update opt value to 6
====================================================
============= 2 th row ============================
oil(r=2, s=0, e=3)
oil(r=2, s=0, e=1)
basecase! return 8
oil(r=2, s=2, e=3)
basecase! return 9
left=8, right=9
@ i=1, updated to 3
@ i=0, updated to 4
@ j=2, updated to 6
max_left=4, max_right=6
update opt value to 10
====================================================

```




{:.output_data_text}

```
10
```


