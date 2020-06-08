---
title: "로봇 청소기"
excerpt: "주어진 조건에 맞게 구현"
categories:
 - algorithms
tags:
 - DFS
 - enumerate
 - simulation
 - samsung
use_math: true
last_modified_at: "2020-05-31"
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
sys.path.append('/home/swyoo/algorithm/')
from sys import stdin
from copy import deepcopy
from utils.verbose import logging_time
from utils.generator import random2D
import numpy as np
```

</div>

# 14503. 로봇 청소기

주어진 조건에 맞게 동작하도록 프로그램을 짜면 된다. 
로봇 청소기는 다음과 같이 작동한다.
<details> <summary> 동작 방법 </summary>

1. 현재 위치를 청소한다.
2. 현재 위치에서 현재 방향을 기준으로 왼쪽방향부터 차례대로 탐색을 진행한다.
    * 왼쪽 방향에 아직 청소하지 않은 공간이 존재한다면, 그 방향으로 회전한 다음 한 칸을 전진하고 1번부터 진행한다.
    * 왼쪽 방향에 청소할 공간이 없다면, 그 방향으로 회전하고 2번으로 돌아간다.
    * 네 방향 모두 청소가 이미 되어있거나 벽인 경우에는, 바라보는 방향을 유지한 채로 한 칸 후진을 하고 2번으로 돌아간다.
    * 네 방향 모두 청소가 이미 되어있거나 벽이면서, 뒤쪽 방향이 벽이라 후진도 할 수 없는 경우에는 작동을 멈춘다.
    
로봇 청소기는 이미 청소되어있는 칸을 또 청소하지 않으며, 벽을 통과할 수 없다.

</details>

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
plot = lambda a: print(np.array(a))
stdin = open('data/robotcleaner.txt')
input = stdin.readline
n, m = list(map(int, input().split()))
i, j, d = list(map(int, input().split()))
grid = [list(map(int, input().split())) for _ in range(n)]
grid
```

</div>




{:.output_data_text}

```
[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
 [1, 0, 0, 0, 1, 1, 1, 1, 0, 1],
 [1, 0, 0, 1, 1, 0, 0, 0, 0, 1],
 [1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
 [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
 [1, 0, 0, 0, 0, 0, 1, 1, 0, 1],
 [1, 0, 0, 0, 0, 0, 1, 1, 0, 1],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
```



## DFS Approach

Using DFS, the algorithm finds `ans`. <br>
Please note that DFS should be call <span style='color:red'>only one neightbor or not!</span> <br>
This is because we have to keep track of only one cleaner robot's moving. <br>

The time complexity is $O(mn)$

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
@logging_time
def solution(grid, i, j, d, show=False):
    mapGo = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    rotate = lambda dx, dy: (-dy, dx)
    isIn = lambda x, y, dx, dy: True if 0 <= x + dx < len(grid) and 0 <= y + dy < len(grid[0]) else False
    forward, ans, cnt = mapGo[d], 0, 2

    def dfs(i, j, fwd):
        nonlocal ans, cnt

        if not grid[i][j]:
            grid[i][j], ans = cnt, ans + 1
            cnt += 1

        for _ in range(4):
            lft = rotate(*fwd)
            if isIn(i, j, *lft) and not grid[i + lft[0]][j + lft[1]]:
                dfs(i + lft[0], j + lft[1], lft)
                return
            else:
                fwd = rotate(*fwd)

        bwd = rotate(*rotate(*fwd))
        if isIn(i, j, *bwd) and grid[i + bwd[0]][j + bwd[1]] != 1:
            dfs(i + bwd[0], j + bwd[1], fwd)

    dfs(i, j, fwd=forward)

    if show: plot(grid), print(ans)
    return ans
```

</div>

Note that `1<cnt` means cleaned orders.

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
solution(deepcopy(grid), i, j, d, show=True, verbose=True)
```

</div>

{:.output_stream}

```
[[ 1  1  1  1  1  1  1  1  1  1]
 [ 1 57 58 47 46 45 44 43 42  1]
 [ 1 56 49 48  1  1  1  1 41  1]
 [ 1 51 50  1  1 37 38 39 40  1]
 [ 1 52  1  1 36 35 32 31  0  1]
 [ 1 53 54 13 12 34 33 30 29  1]
 [ 1 55 15 14 11 10  0  1 28  1]
 [ 1 17 16  3  2  9  1  1 27  1]
 [ 1 18 19  4  5  8  1  1 26  1]
 [ 1 22 20 21  6  7 23 24 25  1]
 [ 1  1  1  1  1  1  1  1  1  1]]
57
WorkingTime[solution]: 0.58389 ms

```




{:.output_data_text}

```
57
```



## Iterative Approach
Consecutive recursive calls lead to big overheads because it has a thread of stack overflow.

Therefore, iterative implementation as follows. 

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
@logging_time
def iterative(grid, i, j, d, show=False):
    mapGo = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    rotate = lambda dx, dy: (-dy, dx)
    isIn = lambda x, y, dx, dy: True if 0 <= x + dx < len(grid) and 0 <= y + dy < len(grid[0]) else False
    forward, ans, cnt = mapGo[d], 0, 2
    while True:
        if not grid[i][j]: grid[i][j], ans, cnt = cnt, ans + 1, cnt + 1  # 1. clean
        cleanAny = False
        for _ in range(4):
            left = rotate(*forward)
            if isIn(i, j, *left) and not grid[i + left[0]][j + left[1]]:  # 2 - [a, b].
                grid[i + left[0]][j + left[1]], ans, cnt = cnt, ans + 1, cnt + 1
                i, j, forward, cleanAny = i + left[0], j + left[1], left, True
                break
            forward = left
        if not cleanAny:  # 2 - [c, d]
            backward = rotate(*rotate(*forward))
            if isIn(i, j, *backward) and grid[i + backward[0]][j + backward[1]] == 1:
                break
            else:
                i, j = i + backward[0], j + backward[1]
                
    if show: plot(grid)
    return ans

print(iterative(deepcopy(grid), i, j, d, show=True, verbose=True))
```

</div>

{:.output_stream}

```
[[ 1  1  1  1  1  1  1  1  1  1]
 [ 1 57 58 47 46 45 44 43 42  1]
 [ 1 56 49 48  1  1  1  1 41  1]
 [ 1 51 50  1  1 37 38 39 40  1]
 [ 1 52  1  1 36 35 32 31  0  1]
 [ 1 53 54 13 12 34 33 30 29  1]
 [ 1 55 15 14 11 10  0  1 28  1]
 [ 1 17 16  3  2  9  1  1 27  1]
 [ 1 18 19  4  5  8  1  1 26  1]
 [ 1 22 20 21  6  7 23 24 25  1]
 [ 1  1  1  1  1  1  1  1  1  1]]
WorkingTime[iterative]: 1.17397 ms
57

```

## Test cases

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
n, m = 1000, 10000
grid = random2D(shape=(n, m), sampling=[0, 1], weights=[0.95, 0.05])
i, j = random.randint(0, n - 1), random.randint(0, m - 1)
grid[i][j] = 0 # start point.
d = random.randint(0, 3) # start direction.
print("start point=({},{}), given direction={}".format(i, j, d))
plot(grid)
print(solution(deepcopy(grid), i, j, d, verbose=True))
print(iterative(deepcopy(grid), i, j, d, verbose=True))
```

</div>

{:.output_stream}

```
start point=(520,9284), given direction=0
[[0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]]
WorkingTime[solution]: 0.55504 ms
238
WorkingTime[iterative]: 0.45419 ms
238

```

## Submitted Code

#### Recursive: DFS

<div class="prompt input_prompt">
In&nbsp;[7]:
</div>

<div class="input_area" markdown="1">

```python
from sys import stdin

stdin = open('data/robotcleaner.txt')  # 제출 시 주석처리
input = stdin.readline
n, m = list(map(int, input().split()))
i, j, d = list(map(int, input().split()))
grid = [list(map(int, input().split())) for _ in range(n)]

def solution(grid, i, j, d):
    mapGo = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    rotate = lambda dx, dy: (-dy, dx)
    isIn = lambda x, y, dx, dy: True if 0 <= x + dx < len(grid) and 0 <= y + dy < len(grid[0]) else False
    forward, ans = mapGo[d], 0

    def dfs(i, j, fwd):
        nonlocal ans

        if not grid[i][j]:
            grid[i][j], ans = 2, ans + 1

        for _ in range(4):
            lft = rotate(*fwd)
            if isIn(i, j, *lft) and not grid[i + lft[0]][j + lft[1]]:
                dfs(i + lft[0], j + lft[1], lft)
                return
            else:
                fwd = rotate(*fwd)

        bwd = rotate(*rotate(*fwd))
        if isIn(i, j, *bwd) and grid[i + bwd[0]][j + bwd[1]] != 1:
            dfs(i + bwd[0], j + bwd[1], fwd)

    dfs(i, j, fwd=forward)
    return ans

print(solution(grid, i, j, d))
```

</div>

{:.output_stream}

```
57

```

#### Iterative

<div class="prompt input_prompt">
In&nbsp;[8]:
</div>

<div class="input_area" markdown="1">

```python
from sys import stdin

stdin = open('data/robotcleaner.txt')  # 제출 시 주석처리
input = stdin.readline
n, m = list(map(int, input().split()))
i, j, d = list(map(int, input().split()))
grid = [list(map(int, input().split())) for _ in range(n)]

def solution(grid, i, j, d):
    mapGo = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    rotate = lambda dx, dy: (-dy, dx)
    isIn = lambda x, y, dx, dy: True if 0 <= x + dx < len(grid) and 0 <= y + dy < len(grid[0]) else False
    forward, ans = mapGo[d], 0
    while True:
        if not grid[i][j]: grid[i][j], ans = 2, ans + 1  # 1. clean
        cleanAny = False
        for _ in range(4):
            left = rotate(*forward)
            if isIn(i, j, *left) and not grid[i + left[0]][j + left[1]]:  # 2 - [a, b].
                grid[i + left[0]][j + left[1]], ans = 2, ans + 1
                i, j, forward, cleanAny = i + left[0], j + left[1], left, True
                break
            forward = left
        if not cleanAny:  # 2 - [c, d]
            backward = rotate(*rotate(*forward))
            if isIn(i, j, *backward) and grid[i + backward[0]][j + backward[1]] == 1:
                break
            else:
                i, j = i + backward[0], j + backward[1]
    return ans


print(solution(grid, i, j, d))
```

</div>

{:.output_stream}

```
57

```

## Report 

구현하는데 어려움이 있었던 점은 다음과 사항들이었다.
1. 2-a 에서 1번으로 되돌아가는 것과 2-b 에서 다시 2로 돌아가야 하는 점이 달라 range(4)를 이용.
2. 새로 시작된 점과 방향을 기준으로 4방향에 대해 모두 막혀있을 경우를 알리는 flag인 `cleanAny`가 필요.

