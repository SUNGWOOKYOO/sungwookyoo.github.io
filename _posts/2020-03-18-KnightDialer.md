---
title: "935.Knight Dialer using python - Leetcode "
excerpt: "find distinct ways to press dial pads"
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
 actions:
  - label: "leetcode"
    url: "https://leetcode.com/problems/knight-dialer/"
  - label: "YouTube"
    url: "https://youtu.be/vjRcT-7b0yA"
---

# 935. Knight Dialer

[leetcode](https://leetcode.com/problems/knight-dialer/)  
[Google bloger](https://hackernoon.com/google-interview-questions-deconstructed-the-knights-dialer-f780d516f029)  
[YouTube](https://youtu.be/vjRcT-7b0yA)  
[discussion](https://leetcode.com/problems/knight-dialer/discuss/190787/How-to-solve-this-problem-explained-for-noobs!!!)

## Problem 
Each time it lands on a key (including the initial placement of the knight), it presses the number of that key, pressing $N$ digits total. <br>
How many **distinct** numbers can you dial in this manner? (see [leetcode](https://leetcode.com/problems/knight-dialer/) problem to know about details) <br>

> it is hard to notice how the knight moves. so, I show you examples. <br>
    at 0, knight can only go to 4 or 6 in the dial below. <br>
    at 1, knight can only go to 6 or 8. <br>
    at 5, knight cannot go anywhere. <br>

I will solve this problem following three approaches.
1. Topdown with memoization
2. Bottom up 
3. Efficient Bottom up

To help your understanding, I provide you with a dial picture as follows.

![](https://assets.leetcode.com/uploads/2018/10/30/keypad.png){:width="300"}

## 1. Topdown with memoization

### Key idea
* Set indices $i, j, n$, positions are $(i, j)$, and the residual depth $n$ to press.
* recursively find the distinct number to press dials.
* merge them and memo.
* enumerate all starting from the position $(i,j,n)$ and merge them.

time complexity: $O(n)$ <br>
space: $O(n)$
> explanation:
if No memo, each pad goes to at most 3. <br>
Therefore, $10 \times 3^n = O(3^n)$ because $3^n$ cases exist for each $(i,j)$, it is higly inefficient. <br>
However, with memo, the algorithm can reuse overlapped subproblems without recursion. <br>
So, the time complexity is reduced to $O(n)$.<br>
This is because the number of entries to memoize is $10n$, each entry takes $O(1)$ (just add the solutions of subproblems).


<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
from collections import defaultdict
MOD = (int)(1e9 + 7)

class Solution(object):    
    def knightDialer(self, N):
        """
        :type N: int
        :rtype: int
        """
        
        memo = defaultdict(tuple) # if not present, None.
        
        def lookup(i, j, n):
            """ enumerate distinct paths from (i, j) position 
                along distinct n numbers. """
            # out of cases
            if (i < 0) or (j < 0) or (i >= 4) or (j >= 3) or ((i == 3) and (j != 1)): 
                # if the knight hops outside of the dial pad, return 0.
                return 0
            
            if n == 1:  # base cases.
                return 1
            
            if memo[(i, j, n)]: # check overlapping subproblems.
                return memo[(i, j, n)]
            
            local = lookup(i - 1, j - 2, n - 1) % MOD + \
                    lookup(i - 2, j - 1, n - 1) % MOD + \
                    lookup(i - 2, j + 1, n - 1) % MOD + \
                    lookup(i - 1, j + 2, n - 1) % MOD + \
                    lookup(i + 1, j + 2, n - 1) % MOD + \
                    lookup(i + 2, j + 1, n - 1) % MOD + \
                    lookup(i + 2, j - 1, n - 1) % MOD + \
                    lookup(i + 1, j - 2, n - 1) % MOD
            
            memo[(i, j, n)] = local # memo
            return local
        
        ans = 0
        for i in range(4):
            for j in range(3):
                ans = (ans + lookup(i, j, N)) % MOD
        
        return ans
        
sol = Solution()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
sol.knightDialer(161)
```

</div>




{:.output_data_text}

```
533302150
```



### 2. Bottom up 

If you think more precisely, we convert index $(i, j)$ to $k$, $k \in [0, 9]$ by preprocessing `neighbors` dictionary.

time complexity: $O(n)$  
space: $O(n)$

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
def knightDialer(N):
    neighbors = {
        0:(4,6),
        1:(6,8),
        2:(7,9),
        3:(4,8),
        4:(0,3,9),
        5:(),
        6:(0,1,7),
        7:(2,6),
        8:(1,3),
        9:(2,4)
    }
    if N == 1: return 10 # naive case
    
    m = defaultdict(lambda: 0) # entries for memo, default value is 0
    for k in range(10):
        m[(1, k)] = 1 # for each pad, path size=1 case.
    
    for i in range(2, N + 1): # path size=i case
        for k in range(10): # for each pad key k
            for j in neighbors[k]:
                m[(i, k)] += m[(i - 1, j)]
        for k in range(10):
            m[(i, k)] %= MOD
    ans = 0
    for k in range(10):
        ans += m[(N, k)] % MOD
    return ans % MOD
    
```

</div>

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
knightDialer(161)
```

</div>




{:.output_data_text}

```
533302150
```



### 3. O(n) time O(1) space DP solution

### Inituition
가능한 neighbors를 dictionary로 만들어 놓고 사용할 수 있다.
또한, `N, N - 1, ..., 1` 까지의 optimal 값을 저장할때,   
`ans` 와 `local` 을 두어 <span style="color:red">redundancy를 최소화 할 수 있어 </span>   
space complexity는 constant가 되고, 
주어진 `N` 에 대해 bottom up 방식으로 `1` 부터 `N` 까지  
`src_key`에서 `dst_key` 까지의 각 pad key별로 local값을 구해 합산해 나가면 정답이 된다.  
이때 걸리는 시간은 `N`$\times$ `10` $\times$ `neighbor 수` = $O(n)$ 이 된다. 
> * reducing redundancy comparing with upper case
    * `ans` entry means $\sum_{:}$`m[:, k]`, where `k in [1, N]`
    * `local` entry means `m[each level, k]`

bottom up approach
[discuss](https://leetcode.com/problems/knight-dialer/discuss/189287/O(n)-time-O(1)-space-DP-solution-%2B-Google-interview-question-writeup)

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
def knightDialer(N):
    # Neighbors maps K: starting_key -> V: list of possible destination_keys
    neighbors = {
        0:(4,6),
        1:(6,8),
        2:(7,9),
        3:(4,8),
        4:(0,3,9),
        5:(),
        6:(0,1,7),
        7:(2,6),
        8:(1,3),
        9:(2,4)
    }
    ans = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # initialization
    for _ in range(N-1):
        local = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for src_key in range(10):
            for dst_key in neighbors[src_key]:
                local[dst_key] = (local[dst_key] + ans[src_key]) % MOD
        ans = local
    return sum(ans) % MOD
```

</div>

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
knightDialer(161)
```

</div>




{:.output_data_text}

```
533302150
```


