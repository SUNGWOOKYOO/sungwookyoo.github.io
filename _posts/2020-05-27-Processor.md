---
title: "프로세서 연결하기"
excerpt: "삼성sw 역량 테스트 문제. 가능한 프로세서를 모두 연결시키되필요한 최소전선수를 찾아라"
categories:
 - algorithms
tags:
 - DFS
 - enumerate
use_math: true
last_modified_at: "2020-05-27"
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
import numpy as np
import random, math
import logging, argparse, yaml, copy
import matplotlib.pyplot as plt
sys.path.append('/home/swyoo/algorithm')
from utils.verbose import logging_time, printProgressBar
from copy import deepcopy
```

</div>

# 1767. 프로세서 연결하기

[삼성 SW link](https://swexpertacademy.com/main/talk/solvingTalk/boardCommuList.do?searchCondition=COMMU_DETAIL-COMMU_TITLE-NICK_NAME_TAG&commuId=AWD_S-56BPoDFAWR&searchKeyword=%ED%94%84%EB%A1%9C%EC%84%B8%EC%84%9C&orderBy=DATE_DESC&pageSize=20&pageIndex=1)

[c++ 로 구현된 좋은 코드](https://2youngjae.tistory.com/117)를 찾아서 이를 바탕으로 python으로 구현하였다.

Notation은 $n \times n$ array 가 주어졌을때, <br>
이 안에 들어있는 processor의 갯수를 $M$ 이라 하자. <br>

각 processor 마다 동,서,남,북,행동X  5가지 cases을 모두 고려해 전선을 까는 경우에 대해 모든 프로세서를 탐색 해보고, <br>
프로세서를 가장 많이 가동 할 수 있으면서 전선이 최소로 필요한 상황에서 최소 전선의 수를 찾는 것이 목표이다.

최악의 경우 모든 cases를 enumerate하는 경우 $O(n5^M)$ 시간이 걸린다. <br>
* 각 프로세서 마다 최대 5번의 방향성을 고려해야 함.
* `inline, drawline` 함수는 $O(n)$ 시간이 걸림.
> 가정상황: $n^2 < 5^M$

하지만, 밑의 방식으로 코딩하면 drawline을 통해 주변 프로세서들이 가능한 방향이 pruning되기 때문에 <br>
최악의 경우와 같이 모두 조사하는 상황은 거의 발생하지 않는다. 

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
pnum = 0  # 찾아야할 파워가 연결된 프로세서 갯수
ans = 1e8  # 찾아야할 사용된 전선들의 최소길이
n = 0
a = []

def isline(r, c, dir):
    global a, n
    if dir == 0:
        for j in range(c + 1, n):
            if a[r][j] != 0:
                return False
    elif dir == 1:
        for j in range(c - 1, -1, -1):
            if a[r][j] != 0:
                return False
    elif dir == 2:
        for i in range(r + 1, n):
            if a[i][c] != 0:
                return False
    elif dir == 3:
        for i in range(r - 1, -1, -1):
            if a[i][c] != 0:
                return False
    return True

def drawline(r, c, dir, fill):
    """ fill=2 means draw
        fill=0 means delete. """
    global a, n
    line = 0
    if dir == 0:
        for j in range(c + 1, n):
            a[r][j] = fill
            line += 1
    elif dir == 1:
        for j in range(c - 1, -1, -1):
            a[r][j] = fill
            line += 1
    elif dir == 2:
        for i in range(r + 1, n):
            a[i][c] = fill
            line += 1
    elif dir == 3:
        for i in range(r - 1, -1, -1):
            a[i][c] = fill
            line += 1
    return line

@logging_time
def solve():
    global a
    def dfs(p, pidx, nump, line):
        """
        p 는 프로세서 인덱싱 정보, pidx는 조사할 프로세서 인덱스,
        nump는 power가 연결된 프로세서 최대 수, line은 조사중인 전선 길이
        """
        global a, pnum, ans, n
        if pidx == len(p):  # 모든 프로세서에 대해 조사 끝났을 경우
            if pnum < nump: # 더 많은 프로세서를 가동할 경우를 찾을 경우
                ans = line
                pnum = nump
            elif pnum == nump: # 프로세서는 이전과 일치하지만, 
                if ans > line: # 더 적은 전선이 필요한 상황을 찾을 경우
                    ans = line
            return
        
        for i in range(4):  # 동, 서, 남, 북 에 대해 조사
            if isline(*p[pidx], i):  # 현재 프로세서를 기준으로 i 방향으로 전선을 까는 것이 가능한지 조사
                # 전선을 drawline을 통해 깔고, 현재 line수를 업데이트한 상태로 다음 process에 대해 조사
                dfs(p, pidx + 1, nump + 1, line + drawline(*p[pidx], dir=i, fill=2))
                drawline(*p[pidx], dir=i, fill=0) # roll back
        dfs(p, pidx + 1, nump, line) # no drawing case

    # dfs 시작지점 지정을 위해 연결되지 않은 프로세서 인덱싱(가장자리는 이미 연결됨)
    p = []
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if a[i][j] == 1:
                p.append((i, j))
    # 가장 자리를 제외하고, 전선을 연결해가며 프로세서 연결이 최대가 될때의 전선 길이 반환
    dfs(p, pidx=0, nump=0, line=0)
```

</div>

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
if __name__ == "__main__":
    sys.stdin = open('data/processor.txt')
    T = int(sys.stdin.readline())

    for tc in range(1, T+1):
        n = int(sys.stdin.readline())
        a = []
        pnum = 0
        ans = 1e8
        for _ in range(n):
            a.append(list(map(int, sys.stdin.readline().split())))
        solve(verbose=True)
        print("#{} {}".format(tc, ans))
```

</div>

{:.output_stream}

```
WorkingTime[solve]: 0.89335 ms
#1 12
WorkingTime[solve]: 9.07135 ms
#2 10
WorkingTime[solve]: 56.23579 ms
#3 24

```

## Review

복습하고자 다시 풀어보았다. 
두 가지 주의 사항이 있었다. 
1. 문제의 조건 <br>
이 문제에서 분명히 확인해야하는 점은 선은 일직선으로만 그을 수 있고, <br>
중간에 어떠한 프로세서나 줄 같은 방해물이 없어야 그 프로세서가 연결 될 수 있다. 
(실수로, 프로세서 끼리 연결되거나 줄끼리 겹치는 경우도 연결되도록 해버려서 한참 헤매었다.) <br>
다음의 함수들이 조건을 만족시키기 위한 핵심 함수들이다.
```python
def go(i, j, ax0, ax1):
    # TODO ...
```
```python
def adj(i, j):
    # TODO ...
```
```python
def marking(i, j, draw, remove=False):
    # TODO ...
```

2. pruning <br>
pruning 하지않으면 시간초과로 통과 할 수 없다. 
따라서, 한 프로세서가 power-on 되지 않는 경우에는 <br> 
아무 선도 연결 하지 않은 case로 몰아서 처리하면, pruning 된다.
``` python
def dfs(...):
    ...
    for isConnect, draw in adj(*pcs[pidx]):
        if isConnect:  # pruning: it is important.
            ...
```

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
import sys
from sys import stdin
import numpy as np
from utils.verbose import logging_time
sys.setrecursionlimit(10000)

@logging_time
def solution(grid):
    ncore, ans, n = 0, 1e10, len(grid)
    pcs = [(i, j) for i in range(1, n - 1) for j in range(1, n - 1) if grid[i][j]]
    up, down, left, right = (-1, 0), (1, 0), (0, -1), (0, 1)

    def go(i, j, ax0, ax1):
        x, y = i + ax0, j + ax1
        draw = []
        while 0 <= x < n and 0 <= y < n:
            if grid[x][y] != 0: return False, []
            draw.append((x, y))
            x, y = x + ax0, y + ax1
        return True, draw

    def adj(i, j):
        for direction in up, down, left, right:
            yield go(i, j, *direction)

    def marking(i, j, draw, remove=False):
        for x, y in draw:
            grid[x][y] = 2 if not remove else 0

    def dfs(pidx, core, line):
        nonlocal ncore, ans
        if pidx == len(pcs):
            if ncore < core:
                ncore, ans = core, line
            elif ncore == core and ans > line:
                ans = line
            return

        for isConnect, draw in adj(*pcs[pidx]):
            if isConnect:  # pruning: it is important.
                marking(*pcs[pidx], draw=draw)
                dfs(pidx + 1, core + 1, line=line + len(draw))
                marking(*pcs[pidx], draw=draw, remove=True)
        dfs(pidx + 1, core, line=line)  # nothing case.

    dfs(0, core=0, line=0)
    return ans

stdin = open('data/processor.txt')
input = stdin.readline

T = int(input())
for t in range(1, T + 1):
    n = int(input())
    grid = [list(map(int, input().split())) for _ in range(n)]
    ans = solution(grid, verbose=True)
    print('#{} {}'.format(t, ans))
```

</div>

{:.output_stream}

```
WorkingTime[solution]: 0.76890 ms
#1 12
WorkingTime[solution]: 8.89230 ms
#2 10
WorkingTime[solution]: 59.32403 ms
#3 24

```

## Test

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
n = 10
grid = np.zeros(shape=(n, n), dtype=int).tolist()
pcs = [(random.randint(0, n - 1), random.randint(0, n - 1)) for _ in range(n)]
for x, y in pcs:
    grid[x][y] = 1
print(np.array(grid))

print(solution(deepcopy(grid), verbose=True))
a = deepcopy(grid)
pnum, ans = 0, 1e8
solve(verbose=True)
ans
```

</div>

{:.output_stream}

```
[[0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 1 0 0 0]
 [1 0 0 0 0 1 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 1 0 0 0]
 [0 0 1 0 0 1 0 0 0 0]
 [0 0 0 0 0 1 0 0 0 0]
 [0 1 1 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 1 0 0]]
WorkingTime[solution]: 20.28561 ms
18
WorkingTime[solve]: 19.77015 ms

```




{:.output_data_text}

```
18
```



## Generate Toy Examples.

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
gridstr = '\n'.join(' '.join(str(e) for e in row) for row in np.array(grid).tolist())
print(gridstr)
```

</div>

{:.output_stream}

```
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 1 0 0 0
1 0 0 0 0 1 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 1 0 0 0
0 0 1 0 0 1 0 0 0 0
0 0 0 0 0 1 0 0 0 0
0 1 1 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 1 0 0

```

<div class="prompt input_prompt">
In&nbsp;[7]:
</div>

<div class="input_area" markdown="1">

```python
grid = [list(map(int, rows.split())) for rows in gridstr.split('\n')]
grid
```

</div>




{:.output_data_text}

```
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
 [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
 [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
 [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]
```



<div class="prompt input_prompt">
In&nbsp;[8]:
</div>

<div class="input_area" markdown="1">

```python
print(solution(deepcopy(grid), verbose=True))
```

</div>

{:.output_stream}

```
WorkingTime[solution]: 20.10751 ms
18

```
