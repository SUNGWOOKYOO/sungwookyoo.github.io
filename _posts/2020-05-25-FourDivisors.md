---
title: "1390.Four Divisors"
excerpt: "practice of a mathmatical algorithms"
categories:
 - algorithms
tags:
 - calculate
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
In&nbsp;[67]:
</div>

<div class="input_area" markdown="1">

```python
from math import sqrt
from typing import List
import random, sys
sys.path.append("/home/swyoo/algorithm/")
from utils.verbose import logging_time
```

</div>

# 1390. Four Divisors

Given an integer array `nums`, <br>
return the sum of divisors of the integers in that array that have exactly four divisors. <br>
If there is no such integer in the array, return `0`.

## Navie

### Idea
Find all distinct divisors. <br>
Enumerate all possible numbers to be divisable.  <br>

Let $n$ be the maximum value of elements in `nums` <br>
Let $m$ be the size of `nums`. <br>

It takes $O(mn)$

This algorithm is too slow to pass the efficiency test for the [leetcode problem](https://leetcode.com/problems/four-divisors/). <br>  
we can improve `findDivs(x)` function, which helps to find all divisors of `x` 

<div class="prompt input_prompt">
In&nbsp;[27]:
</div>

<div class="input_area" markdown="1">

```python
class Solution1:
    @logging_time
    def sumFourDivisors(self, nums: List[int]) -> int:
        """ TLE, too slows. """
        findDivs = lambda x: [i for i in range(1, x + 1) if x % i == 0]
        ans = 0
        for e in nums:
            Divs = findDivs(e)
            if len(Divs) == 4:
                ans += sum(Divs)
        return ans

sol1 = Solution1()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[28]:
</div>

<div class="input_area" markdown="1">

```python
nums = [21,4,7]
print(sol1.sumFourDivisors(nums, verbose=True))
```

</div>

{:.output_stream}

```
WorkingTime[sumFourDivisors]: 0.02003 ms
32

```

# Improved Version

Notice that divisible numbers can be found pairwisely. <br>
e.g., $n = 100$, `(1,100), (2,50), (4,25), (5,20), (10,10)` can be found. 

$2^k = n$, $k=\sqrt n$. <br> 
Therefore, the time complexity can be improved as follows. $O(m\sqrt n)$

<div class="prompt input_prompt">
In&nbsp;[31]:
</div>

<div class="input_area" markdown="1">

```python
class Solution2:
    @logging_time
    def sumFourDivisors(self, nums: List[int]) -> int:
        """ improved """
        def findDivs(x):
            res = []
            for i in range(1, int(sqrt(x)) + 1):
                if x % i == 0:
                    if x / i == i:
                        res.append(i)
                    else:
                        res.extend([i, int(x / i)])
            return res
        ans = 0
        for e in nums:
            Divs = findDivs(e)
            if len(Divs) == 4:
                ans += sum(Divs)
        return ans

sol2 = Solution2()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[32]:
</div>

<div class="input_area" markdown="1">

```python
size = 1000
A = [random.randint(0, 100* size) for i in range(size)]
print(sol1.sumFourDivisors(A, verbose=True))
print(sol2.sumFourDivisors(A, verbose=True))
```

</div>

{:.output_stream}

```
WorkingTime[sumFourDivisors]: 2433.62427 ms
12444942
WorkingTime[sumFourDivisors]: 8.99243 ms
12444942

```

<div class="prompt input_prompt">
In&nbsp;[34]:
</div>

<div class="input_area" markdown="1">

```python
def findDivs(x):
    res = []
    for i in range(1, int(sqrt(x)) + 1):
        if x % i == 0:
            if x / i == i:
                res.append(i)
            else:
                res.extend([i, int(x / i)])
    return res
```

</div>

<div class="prompt input_prompt">
In&nbsp;[36]:
</div>

<div class="input_area" markdown="1">

```python
findDivs(100)
```

</div>




{:.output_data_text}

```
[1, 100, 2, 50, 4, 25, 5, 20, 10]
```



# Referenece
[1] Find all divisors of a natural number | Set 1 [geeksforgeeks](https://www.geeksforgeeks.org/find-divisors-natural-number-set-1/)

<div class="prompt input_prompt">
In&nbsp;[None]:
</div>

<div class="input_area" markdown="1">

```python

```

</div>
