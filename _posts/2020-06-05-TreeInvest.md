---
title: "나무 재테크"
excerpt: "주어진 조건에 따라 시뮬레이션 하는 문제, 적절한 자료구조 사용"
categories:
 - algorithms
tags:
 - datastructure
 - enumerate
 - deque
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
import sys, random, heapq
sys.path.append("/home/swyoo/algorithm/")
from utils.generator import random2D
from utils.verbose import logging_time
from collections import defaultdict, deque
from copy import deepcopy
from typing import DefaultDict, Deque, Tuple, List
from pprint import pprint
import numpy as np
plot = lambda a: print(np.array(a))
```

</div>

# 16235. 나무 재테크

[beakjoon](https://www.acmicpc.net/problem/16235)

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
def generator():
    n = random.randint(1, 10)
    m, k = random.randint(1, n ** 2), random.randint(1, 1000)
    a = random2D(shape=(n, n), randrange=(1, 100))
    data = [(random.randint(0, n - 1), random.randint(0, n - 1), random.randint(1, 10)) for _ in range(m)]
    trees = defaultdict(deque)
    for x, y, z in data:
        trees[(x, y)].append(z)
    for pos, ages in trees.items():
        trees[pos] = deque(sorted(ages))
    ntrs = [[5] * n for _ in range(n)]
    return n, m, k, a, trees, ntrs
n, m, k, a, trees, ntrs = generator()
# print(n ,m, k)
# plot(a)
# print(trees)
```

</div>

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
@logging_time
def solution(a: List[List[int]], trees: DefaultDict[Tuple[int, int], Deque[int]], ntrs: List[List[int]], k: int, show: bool = False):
    n = len(a)
    delta = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def spring_summer():
        # spring
        for (i, j), ages in trees.items():
            alives, deads = deque(), 0
            for age in ages:
                if ntrs[i][j] < age:
                    deads += (age // 2)
                    continue
                alives.append(age + 1)
                ntrs[i][j] -= age
            trees[(i, j)] = alives

            # summer
            ntrs[i][j] += deads

    def fall_winter():
        # fall
        reproduce = []
        for (i, j), ages in trees.items():
            for age in ages:
                if age % 5: continue
                for dx, dy in delta:
                    if 0 <= i + dx < n and 0 <= j + dy < n:
                        reproduce.append((i + dx, j + dy))

        # fall: reproduce
        for pos in reproduce:
            trees[pos].appendleft(1)

        # winter
        for i in range(n):
            for j in range(n):
                ntrs[i][j] += a[i][j]

    for _ in range(k):
        spring_summer()
        if not trees: return 0
        fall_winter()

    return sum(len(vs) for _, vs in trees.items())
```

</div>

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
n, m, k, a, trees, ntrs = generator()
solution(*deepcopy((a, trees, ntrs, k)), show=False, verbose=True)
```

</div>

{:.output_stream}

```
WorkingTime[solution]: 0.18120 ms

```




{:.output_data_text}

```
0
```



## Submitted Code

deque 와 reproduce에서 list사용

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
from collections import defaultdict, deque
from sys import stdin

stdin = open('data/treeinvest.txt')
input = stdin.readline
n, m, k = list(map(int, input().split()))
a = [list(map(int, input().split())) for _ in range(n)]
trees = defaultdict(deque)
for _ in range(m):
    x, y, z = list(map(int, input().split()))
    trees[(x - 1, y - 1)].append(z)
for pos, ages in trees.items():
    trees[pos] = sorted(deque(ages))

ntrs = [[5] * n for _ in range(n)]
delta = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

def solution(a, trees, ntrs, k):

    def spring_summer():
        # spring
        for (i, j), ages in trees.items():
            alives, deads = deque(), 0
            for age in ages:
                if ntrs[i][j] < age:
                    deads += (age // 2)
                    continue
                alives.append(age + 1)
                ntrs[i][j] -= age
            trees[(i, j)] = alives

            # summer
            ntrs[i][j] += deads

    def fall_winter():
        # fall
        reproduce = []
        for (i, j), ages in trees.items():
            for age in ages:
                if age % 5: continue
                for dx, dy in delta:
                    if 0 <= i + dx < n and 0 <= j + dy < n:
                        reproduce.append((i + dx, j + dy))

        # fall: reproduce
        for pos in reproduce:
            trees[pos].appendleft(1)

        # winter
        for i in range(n):
            for j in range(n):
                ntrs[i][j] += a[i][j]

    for _ in range(k):
        spring_summer()
        if not trees: return 0
        fall_winter()

    return sum(len(vs) for _, vs in trees.items())

print(solution(a, trees, ntrs, k))
```

</div>

{:.output_stream}

```
85

```

## Report

시간초과가 뜨는 바람에 많은 시간은 허비했다. <br>

1. 처음 시도했던것은 heap을 사용하였는데, `heappush, heappop` 에 대한 overhead때문에 효율이 좋지 못했다.
2. `deque`를 사용하면, 좀더 효율적으로 어린 나무가 먼저 양분을 먹게 할 수 있다. 
    * 왜냐하면, 봄에 죽은 나무를 제외하고 새로 deque를 만들어 놓고, 가을에 `appendleft` 를 통해 앞쪽에 어린 나무를 추가해나가면 된다.
3. dictionary를 사용한 `items()`가 매우 느린 iteration을 만드는 것같다. 
    * `reproduce`를 dictionart에서 list로 바꾸었더니 pypy3에서 통과되었다. 
