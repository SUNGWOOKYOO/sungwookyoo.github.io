---
title: "인구이동"
excerpt: "Union Find을 사용하여 시뮬레이션 해보는 알고리즘"
categories:
 - algorithms
tags:
 - datastructure
 - union find
use_math: true
last_modified_at: "2020-06-05"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/teaser.jpg
 overlay_filter: 0.5
---

<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
import sys
sys.path.append("/home/swyoo/algorithm/")
from sys import stdin
from utils.verbose import visualize_ds, logging_time
from collections import defaultdict
from statistics import mean
import numpy as np
```

</div>

# 16234. 인구 이동

<details> 
    <summary> 문제 설명 </summary>
        인구 이동은 다음과 같이 진행되고, 
        더 이상 아래 방법에 의해 인구 이동이 없을 때까지 지속된다. 
    <ol> 
        <li> 국경선을 공유하는 두 나라의 인구 차이가 L명 이상, R명 이하라면, <br>
            두 나라가 공유하는 국경선을 오늘 하루동안 연다. </li>
        <li> 위의 조건에 의해 열어야하는 국경선이 모두 열렸다면, 인구 이동을 시작한다. </li>
        <li> 국경선이 열려있어 인접한 칸만을 이용해 이동할 수 있으면, 그 나라를 오늘 하루 동안은 연합이라고 한다. </li>
        <li> 연합을 이루고 있는 각 칸의 인구수는 <br>
            (연합의 인구수) / (연합을 이루고 있는 칸의 개수)가 된다. 편의상 소수점은 버린다. </li>
        <li> 연합을 해체하고, 모든 국경선을 닫는다.</li>
    </ol> 
</details>


**Notations**
1. $n$: 격자의 한줄 크기, 총 격자수는 $n^2$.
2. $L, R$ 인구차이가 이 사이라면 연합 가능!.

## Parse Data

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
stdin = open('data/popshift.txt')
input = stdin.readline
plot = lambda a: print(np.array(a))

n, L, R = list(map(int, input().split()))
a = [list(map(int, input().split())) for _ in range(n)]
plot(a)
```

</div>

{:.output_stream}

```
[[ 10 100  20  90]
 [ 80 100  60  70]
 [ 70  20  30  40]
 [ 50  20 100  10]]

```

## Idea

union find 를 이용하여 푼다: 각 Step 별로 disjoint set 을 구성한다. <br>
1. disjoint set을 구성할때, 각 격자별로 인접한 격자의 값과 비교해서 L, R 사이라면 union한다. <br>
2. djsjoint set이 구성되었다면, 인구 이동을 실행한다. <br>
3. 다음 step을 진행한다.

Time Complexity Analysis

1. 모든 격자의 수는 $n^2$ 이므로 disjoint 구성하는데 $O(\alpha n^2)$
2. 인구의 재배치 $O(n^2)$
3. 인구 재배치가 없을 때(구성된 disjoint set의 representative수 = $n^2$)까지 반복. 

따라서, 인구 이동이 끝날때까지의 총 step을 $k$라고하면 $O(k \alpha n^2)$.



<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
@logging_time
def solution(a, L, R, show=False):

    def find(x):
        if x not in par:
            par[x] = x
            rnk[x] = 0
            return par[x]
        if x != par[x]:
            par[x] = find(par[x])
        return par[x]

    def union(x, y):
        x, y = find(x), find(y)
        if x == y: return
        if rnk[x] > rnk[y]:
            x, y = y, x
        par[x] = y
        if rnk[x] == rnk[y]:
            rnk[y] += 1

    n = len(a)
    ans = 0
    while True:
        par, rnk = {}, {}
        for i in range(n):
            for j in range(n):
                for x, y in [(i + 1, j), (i, j + 1)]:
                    if x < n and y < n and L <= abs(a[i][j] - a[x][y]) <= R:
                        union((i, j), (x, y))
        rpr = set()
        [rpr.add((i, j)) for i in range(n) for j in range(n) if find((i, j)) == (i, j)]
        if len(rpr) == n ** 2: break
        groups = defaultdict(list)
        [groups[par[(i, j)]].append((i, j)) for i in range(n) for j in range(n) if par[(i, j)] in rpr]
        for k, vs in groups.items():
            avg = int(mean([a[i][j] for i, j in vs]))
            for i, j in vs:
                a[i][j] = avg
        prv = len(rpr)
        ans += 1
        if show: plot(a)
    return ans
print(solution(a, L, R, show=True, verbose=True))
```

</div>

{:.output_stream}

```
[[ 10 100  50  50]
 [ 50  50  50  50]
 [ 50  50  50  50]
 [ 50  50 100  50]]
[[30 66 66 50]
 [30 66 50 50]
 [50 50 62 50]
 [50 62 62 62]]
[[48 48 54 54]
 [54 54 54 50]
 [54 54 54 54]
 [54 54 62 54]]
WorkingTime[solution]: 1.06740 ms
3

```

## Submitted code

pypy3로 제출해야 시간초과가 뜨지 않는다. 
좀더 효율적으로 코드를 짤 필요가 있어보인다. 

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
from sys import stdin
from collections import defaultdict
from statistics import mean
stdin = open('data/popshift.txt')
input = stdin.readline

n, L, R = list(map(int, input().split()))
a = [list(map(int, input().split())) for _ in range(n)]
def solution(a, L, R):

    def find(x):
        if x not in par:
            par[x] = x
            rnk[x] = 0
            return par[x]
        if x != par[x]:
            par[x] = find(par[x])
        return par[x]

    def union(x, y):
        x, y = find(x), find(y)
        if x == y: return
        if rnk[x] > rnk[y]:
            x, y = y, x
        par[x] = y
        if rnk[x] == rnk[y]:
            rnk[y] += 1

    n = len(a)
    ans = 0
    while True:
        par, rnk = {}, {}
        for i in range(n):
            for j in range(n):
                for x, y in [(i + 1, j), (i, j + 1)]:
                    if x < n and y < n and L <= abs(a[i][j] - a[x][y]) <= R:
                        union((i, j), (x, y))
        rpr = set()
        [rpr.add((i, j)) for i in range(n) for j in range(n) if find((i, j)) == (i, j)]
        if len(rpr) == n ** 2: break
        groups = defaultdict(list)
        [groups[par[(i, j)]].append((i, j)) for i in range(n) for j in range(n) if par[(i, j)] in rpr]
        for k, vs in groups.items():
            avg = int(mean([a[i][j] for i, j in vs]))
            for i, j in vs:
                a[i][j] = avg
        prv = len(rpr)
        ans += 1
        # plot(a)
    return ans
print(solution(a, L, R))
```

</div>

{:.output_stream}

```
3

```
