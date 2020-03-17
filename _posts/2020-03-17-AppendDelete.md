---
title: "Append and Delete - hackerrank"
excerpt: "make source string to target string using Append and Delete operations"
categories:
 - algorithms
tags:
 - enumerate
use_math: true
last_modified_at: "2020-03-17"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/algorithms/algo.png
 overlay_filter: 0.5
 caption: algorithm
 actions:
  - label: "hackerrank"
    url: "https://www.hackerrank.com/challenges/append-and-delete/problem"
---

<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
import sys
sys.path.append("/home/swyoo/algorithm/")
```

</div>

# Append and Delete
[hackerrank](https://www.hackerrank.com/challenges/append-and-delete/problem)

## Key Idea
Note that the algorithm should delete all characters until reacing LCS(Longest Common Prefix). <br>
And then, append some characters to make the target string.
There are some exceptions.
1. `k` operation is large enough to make target string.
2. residual operations can be used append and delete until making target string. 
    * Note that the number of residual operations should be even.

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
def solve(s, t, k):
    if len(s) + len(t) <= k: return "Yes" # exception.
    i, r = 0, min(len(s), len(t))
    while i < r and s[i] == t[i]: i += 1
    even = len(t) - len(s) + k
    n = len(s) - i + len(t) - i # of total operations.
    if k < n: return "No"
    if k == n: return "Yes"
    return "No" if (k-n)%2 else "Yes" # if residual operations remain.
```

</div>

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
s = "hackerhappy"
t = "hackerrank"
k = 9
```

</div>

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
solve(s, t, k)
```

</div>




{:.output_data_text}

```
'Yes'
```



## Time Complexity
$O(n)$

# Report

`easy` 문제였지만, enumerate하는 과정에서 exception들을 잘 고려야했다.
exception들을 잘 처리하는 것도 중요한 문제이다.
