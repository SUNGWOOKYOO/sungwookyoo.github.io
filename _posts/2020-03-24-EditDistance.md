---
title: "72.Edit Distance using python - Leetcode"
excerpt: "#"
categories:
 - algorithms
tags:
 - string
 - DP
use_math: true
last_modified_at: "2020-03-24"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/algorithms/algo.png
 overlay_filter: 0.5
 caption: algorithm
 actions:
  - label: "geeksforgeeks"
    url: "https://www.geeksforgeeks.org/edit-distance-dp-5/"
  - label: "leetcode"
    url: "https://leetcode.com/problems/edit-distance/"
---
<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
import sys
import numpy as np
from collections import defaultdict
sys.path.append("/home/swyoo/algorithm")
from utils.verbose import logging_time
```

</div>

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
# toy example1
word1 = "intention"
word2 = "execution"
# # toy example2
# word1 = "dinitrophenylhydrazine"
# word2 = "benzalphenylhydrazone"
```

</div>

# 72. Edit Distance

`word1` 를 insert, remove, replace 세 종류의 연산을 통해` word2`로 바꾸는데 필요한 최소 연산 수를 구해라.

## Key Idea
`word1`, `word2`에서 보는 index를 각각 `i,j` 로 두고, <br>
**뒤에서부터 word1과 word2를 보면서 바꿔야하는 연산을 모두 해보자**. <br>
그 중 **최소의 비용을 택해 `return`**해나간다. <br>
`f(i,j)`함수를 `word1[:i+1]`을 `word2[j+1]`로 바꾸는데 필요한 **최소의 연산**을 구하는 함수로 구현해보자.

* 일단 character가 같은 부분은 건드릴 필요없다. 따라서 skip 하고, 다음 recursion을 한다.
```python
if word1[i - 1] == word2[j - 1]: return f(i - 1, j - 1)
```
* `word1`에 대해 `insert`연산을 하여 `word1[i]`뒤에 `word2[j]`와 똑같은 character를 `insert`한다. <br>
    그러면, 다음 recursion에서 봐야할 부분은 `f(i, j-1)` <br>
    (`word1[i+1]`과 `word2[j]`가 같아지니까 나머지 부분들만 수정하면 됨.)
* `word1`에 대해 `remove`연산을 한다면, `word1[i]`를 삭제한다. <br>
    다음 recursion에서 봐야할 부분은 `f(i-1, j)`
* `word1`에 대해 `replace`연산을 한다면, `word1[i]`를 `word2[j]`로 바꾼다. <br>
    다음 recursion에서 봐야할 부분은 `f(i-1, j-1)`

## Time Complexity
memo를 하지 않고 모든 경우를 볼경우 시간복잡도는 $O(3^{mn})$ 이다. <br>
memo를 한다면, memo 해야할 entry수는 $mn$, 각 entry당 $O(1)$ 이므로 시간복잡도는 $O(mn)$ <br>

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
class Solution(object):
    @logging_time
    def minDistance(self, word1, word2, memoization=True):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        m = len(word1)
        n = len(word2)
        memo = dict()
        def f(i, j):
            if memoization and (i, j) in memo:
                return memo[(i,j)]
            if i == 0: return j  # insert j times
            if j == 0: return i
            if word1[i - 1] == word2[j - 1]: return f(i - 1, j - 1)
            insert = f(i, j - 1)
            remove = f(i - 1, j)
            replace = f(i - 1, j - 1)
            loc = 1 + min(insert, remove, replace)
            memo[(i, j)] = loc
            return loc  # take the min case.
        return f(m, n)
    
sol = Solution()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
ans1 = sol.minDistance(word1, word2, memoization=False, verbose=True)
ans2 = sol.minDistance(word1, word2, memoization=True, verbose=True)
assert ans1 == ans2
```

</div>

{:.output_stream}

```
WorkingTime[minDistance]: 0.39744 ms
WorkingTime[minDistance]: 0.03123 ms

```

## Bottom Up

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
class Solution(object):
    @logging_time
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        m = len(word1)
        n = len(word2)
        memo = dict()
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0: memo[(i, j)] = j
                elif j == 0: memo[(i, j)] = i
                elif word1[i-1] == word2[j-1]:
                    memo[(i, j)] = memo[(i-1, j-1)]
                else:
                    insert = memo[(i, j - 1)]
                    remove = memo[(i - 1, j)]
                    replace = memo[(i - 1, j - 1)]
                    memo[(i,j)] = 1 + min(insert, remove, replace)
        return memo[(m, n)]
    
sol = Solution()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
sol.minDistance(word1, word2, verbose=True)
```

</div>

{:.output_stream}

```
WorkingTime[minDistance]: 0.06914 ms

```




{:.output_data_text}

```
5
```



# Report 

모든 entry를 구하며 올라와야 하기 때문에 <br>
bottom up 보다 recursive가 recursion하는 overhead를 감안하더라도 더 빠른경우가 있다. <br>

왜냐하면, recursive는 동일한 character가 많을 수록 skip 하고 넘어가기 때문이다. 
