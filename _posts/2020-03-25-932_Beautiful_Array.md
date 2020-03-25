---
title: "932.Beautiful Array - Leetcode"
excerpt: "using divide and conquer, find any beautiful array. "
categories:
 - algorithms
tags:
 - DivideConquer
use_math: true
last_modified_at: "2020-03-25"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/algorithms/algo.png
 overlay_filter: 0.5
 caption: algorithm
 actions:
  - label: "leetcode"
    url: "https://leetcode.com/problems/beautiful-array/"
---

<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
import sys
sys.path.append("/home/swyoo/algorithm")
from utils.verbose import logging_time
```

</div>

# 932. Beautiful Array

Given N, return any beautiful array A.  (**It is guaranteed that one exists**.) <br>
1, 2, ..., N 까지의 숫자로 만들어진 permutation 조합 중 beautiful array를 찾아 return하면 된다. <br>
array가 beautiful하다는 것은 내부의 element가 다음과 같은 성질 만족하면 된다. <br>
* `For every i < j, there is no k with i < k < j such that A[k] * 2 = A[i] + A[j].`
 > We'll use the term "arithmetic-free" interchangeably with "beautiful".


[leetcode](https://leetcode.com/problems/beautiful-array/)

## Key Idea

핵심 내용은 다음과 같다. <br>
array가 beautiful 이 되려면 array가 1, ... , N으로 순서대로 나열되어 있다는 가정하에<br>
**odd index에 해당하는 부분과 even index에 해당하는 부분에서** 각각 $A_i, A_j$를 가져와야 한다. <br>
> 예를 들면, `[1,3,5,7,9]`와 `[2,4,6,8,10]` 에서 `(1 + 2) / 2, (1 + 4) / 2`는 정수가 아니므로 <br>
이에 해당하는 $A_k$는 절대 존재하지 않는다. 

그래서 `[1,3,5,7,9] + [2,4,6,8,10]` 로 partition을 하면, `i< 절반 인덱스 이상의 j`에 대해서는 beautiful하다. <br>
그런데, `i<j` 에 대해 beautiful해야 하므로 두부분을 merge하여 beautiful 하도록 recursion 방식을 통해 conquer하도록 할 것이다. <br>
여기서 생각하긴 어렵지만 알아야할 성질이있다. <br>
다음 recursion에서도 마찬가지로 left 부분과 right 부분 각각 <br>
**odd index에 해당하는 부분과 even index에 해당하는 부분에서** 각각 $A_i, A_j$를 가져와야 한다 <br>
> 예를 들면, `[1,3,5,7,9]`의 odd index에 해당하는 `[1,5,9]` 와 even index 에 해당하는 `[3,7]` 에서 <br>
`(1 + 3) / 2, (1 + 7) / 2`는 홀수가 아니므로 <br> 
이에 해당하는 $A_k$는 절대 존재하지 않는다. 
마찬가지로, `[2,4,6,8,10]`도 생각이 가능하다. 

따라서, partition을 해놓고, recursive call을 통해 merge 해나가면, 
leaf recursion(base case)에 도달했을때, beautiful 한 성질이 만족된다.
> 그 동안 많이 사용해온 recursive call을 통해 성질을 만족시키고, merge하는 것과 반대의 경우임에 주의. <br>
잘 생각해보면 **quick sort처럼 partition해놓고, recursive call을 통해 sort된 순서를 만드는 방식**과 동일하다. 

Time complexity: $O(nlogn)$

밑의 구현 코드는 reference [[1]](https://leetcode.com/problems/beautiful-array/discuss/187669/Share-my-O(NlogN)-C%2B%2B-solution-with-proof-and-explanation)을 바탕으로 python으로 구현했다.
> odd index와 even index의 정확한 위치를 boolean masking을 통해 구현

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
class Solution(object):
    @logging_time
    def beautifulArray(self, N):
        """
        :type N: int
        :rtype: List[int]
        """
        A = list(range(1, N + 1))

        def partition(s, e, mask):
            """ O(n) """
            i = s
            for j in range(s, e + 1):
                if (A[j] & mask) != 0:
                    A[j], A[i] = A[i], A[j]
                    i += 1
            return i

        def dc(s, e, mask):
            """ O(nlogn)"""
            if s >= e: return
            mid = partition(s, e, mask)
            dc(s, mid - 1, mask << 1)
            dc(mid, e, mask << 1)

        dc(0, N - 1, 1)
        return A
```

</div>

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
sol = Solution()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
N = 10
sol.beautifulArray(N, verbose=True)
```

</div>

{:.output_stream}

```
WorkingTime[beautifulArray]: 0.01550 ms

```




{:.output_data_text}

```
[7, 3, 5, 9, 1, 6, 10, 2, 4, 8]
```



python의 특징을 살린 멋진 [code](https://leetcode.com/problems/beautiful-array/discuss/186660/Python-recursion)가 있었는데 다음과 같다.

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
class Solution:
    @logging_time
    def beautifulArray(self, N):
        A = list(range(1, N + 1))

        def f(tmp):
            if len(tmp) <= 2: return tmp
            return f(tmp[::2]) + f(tmp[1::2])

        return f(A)

sol = Solution()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
N = 10
sol.beautifulArray(N, verbose=True)
```

</div>

{:.output_stream}

```
WorkingTime[beautifulArray]: 0.00739 ms

```




{:.output_data_text}

```
[1, 9, 5, 3, 7, 2, 10, 6, 4, 8]
```



# Reference
[1] [simple to understand](https://leetcode.com/problems/beautiful-array/discuss/187669/Share-my-O(NlogN)-C%2B%2B-solution-with-proof-and-explanation) <br>
[2] [simple but difficult to understand](https://leetcode.com/problems/beautiful-array/discuss/186679/Odd-%2B-Even-Pattern-O(N)) <br>
[3] [fancy python code](https://leetcode.com/problems/beautiful-array/discuss/186660/Python-recursion) 
