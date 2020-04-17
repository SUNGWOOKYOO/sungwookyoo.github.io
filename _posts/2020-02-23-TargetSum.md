---
title: "494.Target Sum"
excerpt: "algoritm practice"
categories:
 - algorithms
tags:
 - DP
 - DFS
 - BFS
 - enumerate
use_math: true
last_modified_at: "2020-03-16"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/teaser.jpg
 overlay_filter: 0.5
 caption: algorithms
 actions:
  - label: "leetcode"
    url: "https://leetcode.com/problems/target-sum/"
---

# 494. Target Sum

[leetcode](https://leetcode.com/problems/target-sum/)  
good [discuss 1](https://leetcode.com/problems/target-sum/discuss/325250/Python-different-soluctions%3A-DFS-BFS-DP)
[discuss 2](https://leetcode.com/problems/target-sum/discuss/455024/DP-IS-EASY!-5-Steps-to-Think-Through-DP-Questions.)

let n be the size of given array.
and, L be the **range of possible summation**.

## Naive 
$ O(2^n) $

## Topdown with memo, or DFS

### Key idea

* when considering all cases of `i-th depth`, only call forward recursion (`i+1`). <br>
    This is because it makes no calling of duplicated subproblems. <br>
* there are $2^k$ cases in the `k-th depth` problem. <br>
    if there are same residual sum to find, in the situation, if we use memoization, <br>
    we only call the recursion once because we can reuse it.
   
Please see the detailed explanation as follows.

![](assets/images/algorithms/TargetSum.PNG){:width="300"}

Therefore, the Time complexity becomes
$ O(nL)$


<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
import time, sys
sys.path.append("/home/swyoo/algorithm/")
from utils.verbose import logging_time
```

</div>

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
from collections import defaultdict

MIN = -1e8
step1, step2 = 0, 0
class Solution(object):
    
    def findTargetSumWays(self, A, S):
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
        @logging_time
        def naive_agent():
            def naive(i, loc):
                global step1
                """ mutable variable self.cnt can be used. 
                - time limited. """
                if i == len(A): # base case.
                    step1 += 1
                    return 1 if loc == S else 0

                case1 = naive(i + 1, loc + A[i])
                case2 = naive(i + 1, loc - A[i])
                return case1 + case2
            return naive(0, 0)
        
        memo = defaultdict(lambda: MIN)
        @logging_time
        def topdown_agent():
            def topdown(i, loc):
                global step2
                """ topdown with memo, this approach can be thought of as DFS with memo. """
                if memo[(i, loc)] != MIN:
                    return memo[(i, loc)]
                
                if i == len(A): # base case
                    step2 += 1
                    return 1 if loc == S else 0

                case1 = topdown(i + 1, loc + A[i])
                case2 = topdown(i + 1, loc - A[i])
                merged = case1 + case2 
                memo[(i, loc)] = merged

                return merged
            return topdown(0, 0)
        
        @logging_time
        def BFS():
            queue = {0: 1} # {key:summation, value:count}
            for e in A:
                tmp = defaultdict(int)
                for loc, cnt in queue.items():
                    tmp[loc + e] += cnt # case1
                    tmp[loc - e] += cnt # case2
                queue = tmp
                # print(queue)
            return queue[S]
        
        @logging_time
        def bottom():
            """ hard to understand """
            if not A:
                return 0
            dic = {A[0]: 1, -A[0]: 1} if A[0] != 0 else {0: 2}
            for i in range(1, len(A)):
                tdic = {}
                for d in dic:
                    tdic[d + A[i]] = tdic.get(d + A[i], 0) + dic.get(d, 0)
                    tdic[d - A[i]] = tdic.get(d - A[i], 0) + dic.get(d, 0)
                dic = tdic
            return dic.get(S, 0)

        sol1, t1 = naive_agent()
        sol2, t2 = topdown_agent()
        sol3, t3 = BFS()
        sol4, t4 = bottom()
        
        print("naive: {}, step1: {}, t1: {:.3f} ms".format(sol1, step1, t1))
        print("memo: {}, step2: {}, t2: {:.3f} ms".format(sol1, step2, t2))
        print("BFS: {}, t3: {:.3f} ms".format(sol1, t3))
        print("bottom: {}, t4: {:.3f} ms".format(sol1, t4))
        
        print(sol1, sol2, sol3, sol4)
        assert sol1 == sol2 == sol3 == sol4
        
        return sol1
```

</div>

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
sys.stdin = open('data/494_TargetSum.txt', 'r')
T = int(sys.stdin.readline()) # int(input())
for tc in range(1, T + 1):
    sol = Solution()
    nums = list(map(int, sys.stdin.readline().split()))
    S = int(sys.stdin.readline())
    ans = sol.findTargetSumWays(nums, S)
    print("#{} {}".format(tc, ans))
```

</div>

{:.output_stream}

```
naive: 6666, step1: 1048576, t1: 431.022 ms
memo: 6666, step2: 852, t2: 4.025 ms
BFS: 6666, t3: 1.220 ms
bottom: 6666, t4: 1.778 ms
6666 6666 6666 6666
#1 6666
naive: 0, step1: 2097152, t1: 436.486 ms
memo: 0, step2: 2660, t2: 6.589 ms
BFS: 0, t3: 2.424 ms
bottom: 0, t4: 3.749 ms
0 0 0 0
#2 0

```

# Report

As you can see, `the number of steps in the memo` case **is more smaller** than `the number of steps in naive` case. <br>

* cases 를 고려할때, i + 1 로 가면서 forward recursion만 한다는 것이 처음에는 생각보다 어려웠다.  <br>
* k 번째 depth에는 $2^k$개의 cases가 존재하는데, 그 중 residual sum이 같은 것들이 있다면, <br>
    한번만 recursion하면 memo를 통해 O(1)에 구할 수 있다. 이 부분이 처음에는 생각하기 어려울수 있다. 

`있다, 없다.` 또는 `사용, 사용하지 않음` 같은 문제에 적용이 가능한 방식이므로 잘 기억해두자. <br>
개인 취향이겠지만, Tartget Sum 값을 0 부터 증가시켜 계산하는것 보다는 residual 값을 argument로 주는게 더 fancy한 것 같다. <br>
