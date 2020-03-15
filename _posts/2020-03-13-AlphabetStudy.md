---
title: "9480.민정이와 광직이의 알파벳 공부"
excerpt: "enumerate all cases, Target Sum"
categories:
 - algorithms
tags:
 - DFS
 - enumerate
 - samsung
use_math: true
last_modified_at: "2020-03-13"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/algorithms/algo.png
 overlay_filter: 0.5
 caption: algorithm
 actions:
  - label: "SW expert Academy"
    url: "https://swexpertacademy.com/main/code/problem/problemSolver.do?contestProbId=AXAdrmW61ssDFAXq"
---

<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
import sys
import numpy as np
import random, math
import logging, argparse, yaml, copy
import matplotlib.pyplot as plt
from collections import deque, defaultdict
sys.path.append("/home/swyoo/algorithm/")
from utils.verbose import logging_time, printProgressBar
```

</div>

# 9480. 민정이와 광직이의 알파벳 공부

[SW expert Academy](https://swexpertacademy.com/main/code/problem/problemSolver.do?contestProbId=AXAdrmW61ssDFAXq)

## Key Idea
    
This approach is thought as DFS search. However, it is same with naive approach. <br>

There are some reasons why.
* Only call forward recursion (visited list is not required because see a node once). 
* Overlapping subproblem may be occur, so memo(pruning) can be used.
    * Memoization is can be used when **all element of the alphabet counter is same at depth `i`** .
    * If we use memoizaion the time complexity become $O(nL^L)$ where `L = 26`. 
    * <span style='color:red'>However, unfortunately, calling memo is rarely used </span>until reaching the last depth.
        * This is because $L^L$ is too large (there are too many states.).
        * In addition, memo cases is exponential
* Given a word, case 1 (used) and case2 (not used) be merged, 

After all, just enumerate all cases.


When implementing, it is hard to think that 

* Only call forward recursion to avoid redundancy.
* Rollback alphabet counter to consider not used case for each step.

![](assets/images/algorithms/Alphabet.png){:width="300"}

### Time complexity
naive case:
$$
O(2^n) 
$$
memoization case:
$$
O(nL^L) 
$$

alpha2cnt 를 dictionary 하나로 모든 recursion을 돌 경우, rollback과정이 필요.

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
@logging_time
def solve(words):
    word2idx = {w: i for i, w in enumerate(words)}
    alpha2idx = lambda x: ord(x) - ord('a')
    alpha2cnt = defaultdict(int)  # {alpha idx: count}
    global ans, step, num_step
    ans, step, num_step = 0, 1, 2**(len(words))

    def naive(i):
        global ans, step, num_step
        """ recursively forward call and visit all cases by storing alphabet-counter. """
        assert 0 <= i <= len(words), "index error! "
        printProgressBar(iteration=step+1, total=num_step, msg='recursion ...', length=50)
        """ visit last index """
        # when visiting the checking point.(last index)
        if i == len(words):
            step += 1
            if sum([1 for cnt in alpha2cnt.values() if cnt > 0]) == 26:
                ans += 1
            return

        """ note that only forward recursion, seen dictionary not needed."""
        # words[i] used recursion.
        for c in words[i]:  # visit - case 1: used.
            alpha2cnt[alpha2idx(c)] += 1
        naive(i + 1)

        # not used case recursion.
        # before visiting update alphabet counter.
        for c in words[i]:  # visit - case 2: not-used.
            alpha2cnt[alpha2idx(c)] -= 1
        naive(i + 1)

    # call first point index.
    # all cases recursively called at once.
    naive(i=0)

    return ans
```

</div>

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
if __name__ == "__main__":
    SEED = 0
    np.random.seed(seed=SEED)
    random.seed(SEED)
    
    sys.stdin = open('./data/Alphabet.txt')
    T = int(sys.stdin.readline())
    time = [0]*T
    for t in range(T):
        n = int(sys.stdin.readline())
        words = []
        for i in range(n):
            words.append(str(sys.stdin.readline().strip()))
        print("words: {}".format(words))
        ans, time[t] = solve(words)
        print(), print("ans: {}, time: {:.3f}ms".format(ans, time[t]))
```

</div>

{:.output_stream}

```
words: ['cozy', 'lummox', 'gives', 'smart', 'squid', 'who', 'asks', 'for', 'job', 'pen']
|██████████████████████████████████████████████████| 100.1 % - recursion ...
ans: 1, time: 83.152ms
words: ['abcdefghi', 'jklmnopqr', 'stuvwxyz', 'zyxwvuts', 'rqponmlkj', 'ihgfedcba']
|██████████████████████████████████████████████████| 101.6 % - recursion ...
ans: 27, time: 1.623ms

```

## Top Down and Memo

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
def solve(words):
    word2idx = {w: i for i, w in enumerate(words)}
    alpha2idx = lambda x: ord(x) - ord('a')
    global step1, step2, num_step
    step1, step2, num_step = 0, 0, 2**(len(words))
    def updatecnt(i, old):
        """ update counter using words[i] """
        new = [0, ] * 26
        for j in range(26):
            new[j] = old[j]
        for c in words[i]:
            new[alpha2idx(c)] += 1

        return tuple(new)

    def check(counter):
        """ check if all elements of counter is upper than 0. """
        for e in counter:
            if e <= 0:
                return False
        return True
    
    def func(i, alpha2cnt):
        """ alpha2cnt is (i-1)-th counter state. """
        assert 0 <= i <= len(words), "index error! "
        global step1, num_step
        """ recursively forward call and visit all cases by storing alphabet-counter. """
        # printProgressBar(iteration=step1+1, total=num_step, msg='recursion_v1 ...', length=50)
        # when visiting the checking point.(last index)
        if i == len(words):  # base case
            # print("case={}, alpha2cnt={}".format(i, step1, alpha2cnt))
            step1 += 1
            res = 1 if check(alpha2cnt) else 0
            return res
        """ note that only forward recursion. 
        현재 상태인 alphacnt에 따라 i + 1 의 답을 얻는다. """
        # words[i] not used recursion.
        case1 = func(i + 1, alpha2cnt)
        # used case recursion.
        # before visiting update alphabet counter.
        case2 = func(i + 1, updatecnt(i, alpha2cnt))
        return case1 + case2
    
    memo = {}
    def func_v2(i, alpha2cnt):
        """ alpha2cnt is (i-1)-th counter state. """
        assert 0 <= i <= len(words), "index error! "
        global step2, num_step
        """ if memo exist, use it. """
        printProgressBar(iteration=step2+1, total=num_step, msg='recursion_v2 ...', length=50)
        if (i, alpha2cnt) in memo:
            print("memo[({}, {})] used".format(i, alpha2cnt))
            return memo[(i, alpha2cnt)]

        """ recursively forward call and visit all cases by storing alphabet-counter. """
        # when visiting the checking point.(last index)
        if i == len(words):  # base case
            # print("case={}, alpha2cnt={}".format(step2, alpha2cnt))
            step2 += 1
            res = 1 if check(alpha2cnt) else 0
            return res

        """ note that only forward recursion. 
        현재 상태인 alphacnt에 따라 i + 1 의 답을 얻는다. """
        # words[i] not used recursion.
        case1 = func_v2(i + 1, alpha2cnt)
        # used case recursion.
        # before visiting update alphabet counter.
        case2 = func_v2(i + 1, updatecnt(i, alpha2cnt))
        memo[(i, alpha2cnt)] = case1 + case2
        return case1 + case2
    
    @logging_time
    def simple():
        # call first point index.
        return func(i=0, alpha2cnt=(0,) * 26)
    
    @logging_time
    def dynamic():
        return func_v2(i=0, alpha2cnt=(0,) * 26)
    
    ans1, t1 = simple()
    print()
    ans2, t2 = dynamic()
    print()
    # print(memo.items())
    print("ans1: {}, step1: {}, t1: {:.3f}ms".format(ans1, step1, t1))
    print("ans2: {}, step2: {}, t2: {:.3f}ms".format(ans2, step2, t2))
    
    assert ans1 == ans2, "not correct"
    return ans
```

</div>

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
sys.stdin = open('./data/Alphabet.txt')
T = int(sys.stdin.readline())
t1, t2 = [0]*T, [0]*T 
for t in range(1, T + 1):
    print("{}\n\t\ttest{}\n{}".format("="*50, t, "="*50))
    n = int(sys.stdin.readline())
    words = []
    for i in range(n):
        words.append(str(sys.stdin.readline().strip()))
    print("words: {}".format(words))
    ans = solve(words)
```

</div>

{:.output_stream}

```
==================================================
		test1
==================================================
words: ['cozy', 'lummox', 'gives', 'smart', 'squid', 'who', 'asks', 'for', 'job', 'pen']

|██████████████████████████████████████████████████| 100.0 % - recursion_v2 ...
ans1: 1, step1: 1024, t1: 2.898ms
ans2: 1, step2: 1024, t2: 77.757ms
==================================================
		test2
==================================================
words: ['abcdefghi', 'jklmnopqr', 'stuvwxyz', 'zyxwvuts', 'rqponmlkj', 'ihgfedcba']

memo[(4, (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1))] used
memo[(5, (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0))] used
memo[(5, (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))] used
memo[(4, (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))] used
memo[(5, (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2))] used
memo[(4, (1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1))] used
memo[(5, (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0))] used
memo[(5, (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))] used
memo[(4, (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))] used
memo[(5, (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2))] used
|████████████████████████████----------------------| 56.2 % - recursion_v2 ...
ans1: 27, step1: 64, t1: 0.265ms
ans2: 27, step2: 36, t2: 1.366ms

```

test1에서는 메모의 사용이 한번도 발생하지 않았고, test2 에서는 메모가 어느정도 사용되었으나 많이 사용되지는 않는다.

## Anoter data

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
SEED = 2
np.random.seed(seed=SEED)
random.seed(SEED)
for k in range(1, 6):
    print("{}\n\t\ttest{}\n{}".format("="*50, k, "="*50))
    size = 6
    words = [0]*size
    for i in range(size):
        wlen = random.randint(1, 5)
        word = ''.join([chr(random.randint(ord('a'), ord('z'))) for _ in range(wlen)])
        words[i] = word
    print(words) 
    solve(words)
```

</div>

{:.output_stream}

```
==================================================
		test1
==================================================
['c', 'l', 'xz', 'itg', 'bsvfn', 'zxql']

|██████████████████████████████████████████████████| 100.0 % - recursion_v2 ...
ans1: 0, step1: 64, t1: 0.221ms
ans2: 0, step2: 64, t2: 16.487ms
==================================================
		test2
==================================================
['oqiba', 'okm', 'qfrf', 'ha', 'kf', 'qq']

|██████████████████████████████████████████████████| 100.0 % - recursion_v2 ...
ans1: 0, step1: 64, t1: 0.179ms
ans2: 0, step2: 64, t2: 1.521ms
==================================================
		test3
==================================================
['qvr', 'oz', 'xqyl', 'llofy', 'wxou', 'hpipq']

|██████████████████████████████████████████████████| 100.0 % - recursion_v2 ...
ans1: 0, step1: 64, t1: 0.198ms
ans2: 0, step2: 64, t2: 1.521ms
==================================================
		test4
==================================================
['zlvoo', 'sxr', 'pvhk', 'ti', 'jjzw', 'rqqut']

|██████████████████████████████████████████████████| 100.0 % - recursion_v2 ...
ans1: 0, step1: 64, t1: 0.198ms
ans2: 0, step2: 64, t2: 1.496ms
==================================================
		test5
==================================================
['njxgp', 'lvtcz', 'xag', 'b', 'ubish', 'y']

|██████████████████████████████████████████████████| 100.0 % - recursion_v2 ...
ans1: 0, step1: 64, t1: 0.173ms
ans2: 0, step2: 64, t2: 16.439ms

```

<div class="prompt input_prompt">
In&nbsp;[7]:
</div>

<div class="input_area" markdown="1">

```python
26**26 # too large
```

</div>




{:.output_data_text}

```
6156119580207157310796674288400203776
```



# Report 

DFS 또는 recursion을 연습해볼 수 있는 문제였다. <br>
일반적인 DFS의 방식은 adjacent 리스트를 설정하고, 그것을 바탕으로 한번 방문한 것은 또 방문하지 않도록 marking하면서 갈 수 있는 최대 깊이 만큼 가는 방식이다. <br>
하지만, 이 문제에선 잘못된 방향으로 생각할 수 있다. <br>
1. 순서가 고려되면 안된다. DFS를 통해 seaching할때 순서에 따라 다르게 생각하여 count하면 똑같은 경우를 반복해서 체크하게 되는 문제가 있다. <br> 
    이를 피하기위해 이 경우엔 adjacency list를 바로 다음 word에 대해서만 call한다. 그리고, forward recursion만을 하여 basecase가 마지막 index가 되도록한다. <br>
    그래야 redundancy가 생기지않고, 마지막에 모든 character가 있는지 체크할 수 있다. <br>
    이 사실을 생각치 못했기에 내가 했던 비 효율적인 방식은 adjacency list 를 현재 노드(인덱스) 만 빼고 모두 만들었다. 그렇게 하면 redundancy가 매우 많이 생겨 현저히 느려진다. <br>
    또한, 그 redundancy가 생김으로 인해 똑같은 조합을 반복해서 체크하여 카운트 하게 되는것을 방지하기 위해 <br>
    chacter조합을 만들기위해 사용한 모든 word들 의 index에 대해 seen dictionary를 만들었는데, 이 또한 매우 비효율적인 생각이다.
2. 모든 case에 대해 alphabet 을 얼마나 가지고있는지 따져보기 위해서는 단순히 방문하는 것이 아닌, rollback하는 과정이 필요하다.  <br>
    여기서 어려운 부분은 finish 하기 전에 update 되었던 alphabet정보를 **되돌리고, recursion을 다시 call하는 것** 을 생각하기 어려웠다. <br>
    그렇게 하므로써 한 word를 사용한 경우(O) 와 사용하지않은 경우(X) 모두를 고려하게 된다. <br>
    [WordSearch](https://sungwookyoo.github.io/algorithms/WordSearch/)와 유사하지만 다르다는 것을 주목하자. <br>
    (WordSearch 는 순서가 중요하며 visited(seen) 를 update 해나가며 방문하는데 그 과정에서 막다른 길에 갔을 경우 rollack하기 위해 seen을 삭제했다.)
3. 또한, [Target Sum](https://sungwookyoo.github.io/algorithms/TargetSum/) 문제와 거의 유사하다. <br>
    Target Sum 문제는 Target Sum 하나가 정확히 일치해야하지만 여기서는 26개의 counter가 있으며, 값은 0~26 사이이다.
    Target Sum은 두개의 case에 대해 merge가 되며 subproblem들이 residual range 안에서 overlapping 되지만, 
    이 문제는 26 character의 counter의 counter값이 overlapping되었을 때만 memo를 통한 dynamic programming을 사용할 수 있다.
    따라서, 메모량이 매우 많고, 잘 사용되지도 않기때문에 효율이 좋지 못하다. 
    
결국엔, alphabet 셋이 모두 있는지 확인 하려면 대부분 depth 끝까지 recursion해야하고, prunning되는 것들이 거의 없어 대부분의 cases를 enumerate하게 된다. 
