---
title: "494.Target Sum"
excerpt: "algoritm practice"

categories:
  - algorithms
tags:
  - DP
use_math: true
last_modified_at: 2020-02-23
---

[leetcode](https://leetcode.com/problems/target-sum/)  
[good discuss](https://leetcode.com/problems/target-sum/discuss/325250/Python-different-soluctions%3A-DFS-BFS-DP)

let n be the size of given array.
and, L be the range of possible summation.

### Naive 
$ O(2^n) $

### Topdown with memo, or DFS
$ O(nL)$

### python implementations
```python
import time, sys

def logging_time(fn):
    def wrapper_fn(*args, **kwargs):
        start = time.time()
        ans = fn(*args, **kwargs)
        elapsed = (time.time() - start) * 1e3
        print("WorkingTime[{}]: {:.5f} ms".format(fn.__name__, elapsed))
        return ans
    return wrapper_fn
    
from collections import defaultdict

MIN = -1e8
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
                """ mutable variable self.cnt can be used. 
                - time limited. """
                if i == len(A): # base case.
                    return 1 if loc == S else 0

                case1 = naive(i + 1, loc + A[i])
                case2 = naive(i + 1, loc - A[i])
                return case1 + case2
            return naive(0, 0)
        
        memo = defaultdict(lambda: MIN)
        @logging_time
        def topdown_agent():
            def topdown(i, loc):
                """ topdown with memo, this approch can be thought of as DFS with memo.
                - time limited. """
                if i == len(A): # base case
                    memo[(i, loc)] = 1 if loc == S else 0
                    return memo[(i, loc)]

                if memo[(i, loc)] != MIN:
                    return memo[(i, loc)]

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
            
        sol1 = naive_agent()
        sol2 = topdown_agent()
        sol3 = BFS()
        sol4 = bottom()
        
        print(sol1, sol2, sol3, sol4)
        assert sol1 == sol2 == sol3 == sol4
        
        return sol1
```
