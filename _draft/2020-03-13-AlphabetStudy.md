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

[SW expert Academy](https://swexpertacademy.com/main/code/problem/problemSolver.do?contestProbId=AXAdrmW61ssDFAXq)

## Key Idea
    
This approach is thought as DFS search. However, it is same with naive approach. <br>
There are some reasons why.
* only call forward recursion (visited list is not required because see a node once). 
* Also, overlapping subproblem does not occur. 
* Finally, puning cannot be used until reaching the last depth.
* Given a word, case 1 (used) and case2 (not used) cannot be merged. memo cases is exponential

After all, just enumerate all cases.

When implementing, it is hard to think that 

* only call forward recursion to avoid redundancy.
* rollback alphabet counter to consider not used case for each step before return.

![](/assets/images/algorithms/Alphabet.png){:width="300"}

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
from utils.verbose import logging_time, printProgressBar
from collections import deque, defaultdict

VERBOSE = True
# setup logging
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if VERBOSE else logging.WARNING)

@logging_time
def solve(words):
    word2idx = {w: i for i, w in enumerate(words)}
    alpha2idx = lambda x: ord(x) - ord('a')
    alpha2cnt = defaultdict(int)  # {alpha idx: count}
    global ans, step, num_step
    ans, step, num_step = 0, 1, 2**(len(words)+1)

    def naive(i):
        global ans, step, num_step
        """ recursively forward call and visit all cases by storing alphabet-counter. """
        assert 0 <= i <= len(words), "index error! "
        printProgressBar(iteration=step+1, total=num_step, msg='recursion ...', length=50)
        step += 1
        """ visit last index """
        # when visiting the checking point.(last index)
        if i == len(words):
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
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
if __name__ == "__main__":
    SEED = 0
    np.random.seed(seed=SEED)
    random.seed(SEED)
    
    sys.stdin = open('input.txt')
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
|██████████████████████████████████████████████████| 100.0 % - recursion ...
ans: 1, time: 126.181ms
words: ['abcdefghi', 'jklmnopqr', 'stuvwxyz', 'zyxwvuts', 'rqponmlkj', 'ihgfedcba']
|██████████████████████████████████████████████████| 100.0 % - recursion ...
ans: 27, time: 1.971ms

```

# Report 

DFS를 연습하기 아주 좋은 문제였다. <br>
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
3. 또한, [Target Sum](https://sungwookyoo.github.io/algorithms/TargetSum/) 문제와 유사해 보이지만, 다르다. 
    Target Sum은 두개의 case에 대해 merge가 되며 subproblem들이 overlapping 되지만, 
    이 문제는 character의 counter가 overlapping되지 않아 dynamic programming을 사용할 수 없다.
    
결국엔, alphabet 셋이 모두 있는지 확인 하려면 depth 끝까지 recursion해야하고, prunning되는 것들이 하나도 없어 모든 cases를 enumerate하게 된다. 

이 문제를 `Target Set`문제라고 생각할수 있을 거 같다.
