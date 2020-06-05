---
title: "아기상어"
excerpt: "우선순위에 따른 BFS 연습을 하기 좋은 문제"
categories:
 - algorithms
tags:
 - datastructure
 - heap
 - BFS
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
import sys, random
sys.path.append("/home/swyoo/algorithm/")
from sys import stdin
import numpy as np
from heapq import heappush, heappop
from binarytree import build
from utils.verbose import logging_time
from utils.generator import random2D
```

</div>

# 16236. 아기상어

Idea: 
> 우선순위를 바탕으로, 탐색을 해야한다. <br>
즉, queue에서 pop할때 우선순위에 따라 검색할 인덱스가 나와야한다. <br>
따라서, `priority queue`, 즉 `heap`을 사용하자.

1. heap을 사용하여, (distance, row index, col index)를 key값으로 배열을 가지고 있는다. 
2. heap을 바탕으로 우선순위에의한 BFS로 먹을 수 있는 물고기들을 탐색한다. 
3. 먹을 수 있다면, 주어진 조건에 맞는 action을 취한다. 

<details> <summary> Notification </summary>
    <ul>
        <li> <p> 시작점에서, 그리고 물고기를 먹을 때마다 `a`행렬에서 index 해당하는 값을 지우고, <br>
            <code>body, eat, d, seen</code> 등을 업데이트해야한다. </p>
        </li>
        <li> adjacent list의 순서는 중요하지 않다. <br>
            왜냐하면 push, pop할때 <b>heap property</b>를 만족하도록 update되기 때문이다. 
        </li> 
        <li> 잊버리기 쉬운데, 물고기를 먹었으면, <code>seen</code>을 초기화하여 방문했던 grid를 또 방문가능하게 해야한다.
        </li>   
    </ul>
</details>

Time complexity: 

전체 grid의 shape를 $n \times n$, 물고기를 먹은 횟수가 $k$번이라고 하면, <br>
한번 먹기위해서 BFS를 전체 grid에 대해서 수행해야한다. <br>
그런데, queue에 넣고 뺄때마다 우선순위가 유지 되어야하므로 heap을 사용하면, $logn$ 만큼 수행된다. <br>
따라서, $O(k n^2 logn)$

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
stdin = open('data/babyshark.txt')
input = stdin.readline
plot = lambda a: print(np.array(a))

n = int(input())
a = [list(map(int, input().split())) for _ in range(n)]
plot(a)

def visualize(x, body, index=False, additional=False):
    edible = [(d, r, c) for d, r, c in x if 0 < a[r][c] < body]
    if not edible: return  # x contains nothing.
    dists, rows, cols = list(zip(*edible))
    build(dists).pprint(index=index)
    if additional:
        print("{}".format(edible))
```

</div>

{:.output_stream}

```
[[5 4 3 2 3 4]
 [4 3 2 3 4 5]
 [3 2 9 5 6 6]
 [2 1 2 3 4 5]
 [3 2 1 6 5 4]
 [6 6 6 6 6 6]]

```

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
@logging_time
def solution(a, show=False):
    n = len(a)
    pq = [(0, i, j) for i in range(n) for j in range(n) if a[i][j] == 9]
    a[pq[0][1]][pq[0][2]] = 0
    body, eat, ans = 2, 0, 0

    def adj(r, c, body):
        for x, y in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if 0 <= x < n and 0 <= y < n and a[x][y] <= body:
                yield x, y

    seen = set()
    visited, cnt = [[0] *n for _ in range(n)], 0
    while pq:
        d, r, c = heappop(pq)
        if 0 < a[r][c] < body:
            eat, a[r][c], ans = eat + 1, 0, ans + d
            if body == eat:
                body, eat = body + 1, 0
            seen, pq, d = set(), [], 0
            if show:
                cnt += 1
                visited[r][c] = cnt

        for x, y in adj(r, c, body):
            if (x, y) not in seen:
                seen.add((x, y))
                heappush(pq, (d + 1, x, y))

    if show: plot(visited)
    return ans
print(solution(a, show=False, verbose=True))
```

</div>

{:.output_stream}

```
WorkingTime[solution]: 0.47588 ms
60

```

## Test Cases

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
n = 10000
a = random2D(shape=(n, n), randrange=(0, 8))
a[random.randint(0, n - 1)][random.randint(0, n - 1)] = 9
# plot(a)
print(solution(a, show=False, verbose=True))
```

</div>

{:.output_stream}

```
WorkingTime[solution]: 6470.71433 ms
7

```

## Submmitted Code

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
from sys import stdin
from heapq import heappush, heappop

stdin = open('data/babyshark.txt')
input = stdin.readline

n = int(input())
a = [list(map(int, input().split())) for _ in range(n)]

def solution(a):
    n = len(a)
    pq = [(0, i, j) for i in range(n) for j in range(n) if a[i][j] == 9]
    a[pq[0][1]][pq[0][2]] = 0
    body, eat, ans = 2, 0, 0

    def adj(r, c, body):
        for x, y in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if 0 <= x < n and 0 <= y < n and a[x][y] <= body:
                yield x, y

    seen = set()
    while pq:
        d, r, c = heappop(pq)
        if 0 < a[r][c] < body:
            eat, a[r][c], ans = eat + 1, 0, ans + d
            if body == eat:
                body, eat = body + 1, 0
            seen, pq, d = set(), [], 0

        for x, y in adj(r, c, body):
            if (x, y) not in seen:
                seen.add((x, y))
                heappush(pq, (d + 1, x, y))
    return ans

print(solution(a))
```

</div>

{:.output_stream}

```
60

```

## Reference

[1] [Beakjoon 아기상어](https://www.acmicpc.net/problem/16236) <br>
[2] [rebas's blog](https://rebas.kr/714)
