---
title: "구슬탈출2"
excerpt: "samsung sw test practice, DFS and BFS practice"
categories:
 - algorithms
tags:
 - BFS
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
In&nbsp;[22]:
</div>

<div class="input_area" markdown="1">

```python
import sys, random
from sys import stdin
from collections import deque
from copy import deepcopy
import numpy as np
sys.setrecursionlimit(10000)
sys.path.append("/home/swyoo/algorithm/")
from utils.verbose import logging_time
```

</div>

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
def visualize(s, i, j, p, r):
    tmp = deepcopy(s)
    tmp[i][j] = 'R'
    tmp[p][r] = 'B'
    print(np.array(tmp))
```

</div>

# 13460. 구슬 탈출 2
Let $m,n$ be the shape of a given array. <br>
Let the number of maximum walk be $N$, where $N \le 10$. <br>
Note that 4 cases exist(north, south, east, west) for each walk. <br>
Therefore, enumerate all cases: $O(max(m,n)4^N)$.

Toy example generater is implemented as follows. <br>
(Please beware that `R, O, B` can be overlapped, in this case, regenerate the sample.)

<div class="prompt input_prompt">
In&nbsp;[20]:
</div>

<div class="input_area" markdown="1">

```python
n, m = 5, 4
s = [['#'] * m for _ in range(n)]
for _ in range(1, n - 1):
    for _ in range(1, m - 1):
        s[random.randint(1, n - 2) ][random.randint(1, m - 2) ] = '.'
s[random.randint(1, n - 2) ][random.randint(1, m - 2)] = 'B'
s[random.randint(1, n - 2) ][random.randint(1, m - 2)] = 'R'
s[random.randint(1, n - 2) ][random.randint(1, m - 2)] = 'O'
s
```

</div>




{:.output_data_text}

```
[['#', '#', '#', '#'],
 ['#', '#', 'B', '#'],
 ['#', '.', 'O', '#'],
 ['#', '.', 'R', '#'],
 ['#', '#', '#', '#']]
```



## DFS Approach

1. There are two seeds, which are red and blue balls. <br>

 We have to consider these two balls at the same time. 
 First, find the first positions of two balls. <br>
 Second, keep track of two balls by recursive call! <br>
 > where adjacent list implementation is important!.

2. There are some trick parts when implementing adjacent list. <br>

 **At first**, go until come acrossing not `'.'`. <br>
 **Secondly**, if next walk is `'O'`, go there. <br>
 However, overlapping two balls is possilbe. So, **the exceptions should be handled**. <br>
 <span style="color:red">**How to handle the exceptions?**</span> I refered [this article](https://sangdo913.tistory.com/162)
 <details> <summary> Details </summary> 
    <p> Remove initial points of 'R' and 'B', and store it. <br>
        The algorithm will keep track of the positions of 'R'and 'B' <br>
        However, the positions of 'R' and 'B' can be overlapped. <br>
        For preventing the overlapping circumstance, the algorithm <b>count the number of walks!</b>. <br>
        If the algorithm knows how many the ball walks, we can notify the right position. <br> 
        E.g., <center><code>#RB.....#</code></center> <br>
        Imagine that both balls move to right-side direction. <br>
        After moving, the positions will become as follows. <br> 
        <center><code>#.....RB#</code></center> <br>
        <b>The number of steps of marble 'R' is greater than the number of steps of marble 'B'. </b><br>
        Therefore, a ball with a small number of steps must precede the other ball <br>
        by 1 step in the direction of movement.
    </p> 
 </details>

```python
def adj(i, j, p, r):
    for x, y in [(up, 0), (down, 0), (0, right), (0, left)]:
        # go until come acrossing '#' or 'O'.
        nextred, walkred = go(i, j, row=x, col=y)
        nextblue, walkblue = go(p, r, row=x, col=y)
        # Exception handing: overlapping circumstances.
        if nextred == nextblue and s[nextred[0]][nextred[1]] != 'O':
            if walkred < walkblue:
                nextblue = (nextblue[0] - x, nextblue[1] - y)
            else:
                nextred = (nextred[0] - x, nextred[1] - y)
        if (i, j, p, r) != (*nextred, *nextblue):
            # Prune if both balls do not move.
            yield (*nextred, *nextblue, z)
```

### Navie DFS 

Enumerate all cases: $O(max(m,n)4^N)$. <br>
Pruning can be operated, but inefficient!.

<div class="prompt input_prompt">
In&nbsp;[141]:
</div>

<div class="input_area" markdown="1">

```python
up = left = -1
down = right = 1
INF = 1e20

@logging_time
def solution1(s):
    n, m = len(s), len(s[0])
    red = [-1, -1]
    blue = [-1, -1]
    for i in range(n):
        for j in range(m):
            if s[i][j] == 'R':
                red = [i, j]
                s[i][j] = '.'
            elif s[i][j] == 'B':
                blue = [i, j]
                s[i][j] = '.'

    def go(i, j, row, col):
        x, y = i, j
        walk = 0
        while s[x + row][y + col] == '.':
            x, y = x + row, y + col
            walk += 1
        if s[x + row][y + col] == 'O':
            x, y = x + row, y + col
            walk += 1
        return (x, y), walk

    def adj(i, j, p, r):
        for x, y in [(up, 0), (down, 0), (0, right), (0, left)]:
            nextred, walkred = go(i, j, row=x, col=y)
            nextblue, walkblue = go(p, r, row=x, col=y)
            if nextred == nextblue and s[nextred[0]][nextred[1]] != 'O':
                if walkred < walkblue:
                    nextblue = (nextblue[0] - x, nextblue[1] - y)
                else:
                    nextred = (nextred[0] - x, nextred[1] - y)
            if (i, j, p, r) != (*nextred, *nextblue):  # Prune if both balls do not move.
                yield (*nextred, *nextblue)

    ans = INF
    
    def dfs(i, j, p, r, cnt=0):
        nonlocal ans

        if cnt > 10 or s[p][r] == 'O':
            return

        if s[i][j] == 'O':
            ans = min(ans, cnt)
            return

        for ii, jj, pp, rr in adj(i, j, p, r):
            dfs(ii, jj, pp, rr, cnt + 1)

    dfs(*red, *blue, cnt=0)
    return ans if ans != INF else -1

print(solution1(deepcopy(s), verbose=True))
```

</div>

{:.output_stream}

```
WorkingTime[solution1]: 56.03266 ms
3

```

### DFS with `seen`

With `seen`, the algorithm do not repeat the path that has passed. <br>
However, DFS visted order is followed by the order of adjacent list for each nodes. <br> 
Please note that `ans` is updated not greedly. <br> 
It means reverting `seen` is needed. <br>
This is because it is possible to exist shorter paths. <br>
Therefore, `seen` should be marked for both balls' positions **before calling**, and revert it **after return**.

Refer these three problems(Pratice of backtracking(or pruning)). 
* [problem 1 - Connect Processors](https://sungwookyoo.github.io/algorithms/Processor/)
* [problem 2 - Word Search](https://sungwookyoo.github.io/algorithms/WordSearch/)
* [problem 3 - N Queens](https://sungwookyoo.github.io/algorithms/51_NQueens/)

<div class="prompt input_prompt">
In&nbsp;[136]:
</div>

<div class="input_area" markdown="1">

```python
up = left = -1
down = right = 1
INF = 1e20

@logging_time
def solution1(s):
    n, m = len(s), len(s[0])
    red = [-1, -1]
    blue = [-1, -1]
    for i in range(n):
        for j in range(m):
            if s[i][j] == 'R':
                red = [i, j]
                s[i][j] = '.'
            elif s[i][j] == 'B':
                blue = [i, j]
                s[i][j] = '.'

    def go(i, j, row, col):
        x, y = i, j
        walk = 0
        while s[x + row][y + col] == '.':
            x, y = x + row, y + col
            walk += 1
        if s[x + row][y + col] == 'O':
            x, y = x + row, y + col
            walk += 1
        return (x, y), walk

    def adj(i, j, p, r):
        for x, y, z in [(up, 0, 'up'), (down, 0, 'down'), (0, right, 'right'), (0, left, 'left')]:
            nextred, walkred = go(i, j, row=x, col=y)
            nextblue, walkblue = go(p, r, row=x, col=y)
            if nextred == nextblue and s[nextred[0]][nextred[1]] != 'O':
                if walkred < walkblue:
                    nextblue = (nextblue[0] - x, nextblue[1] - y)
                else:
                    nextred = (nextred[0] - x, nextred[1] - y)
            if (i, j, p, r) != (*nextred, *nextblue):  # Prune if both balls do not move.
                yield (*nextred, *nextblue, z)

    ans = INF
    seen = set()

    def dfs(i, j, p, r, cnt=0):
        nonlocal ans

        if cnt > 10 or s[p][r] == 'O':
            return

        if s[i][j] == 'O':
            ans = min(ans, cnt)
            return

        for ii, jj, pp, rr, z in adj(i, j, p, r):
            if (ii, jj, pp, rr) not in seen:
                seen.add((ii, jj, pp, rr))
                dfs(ii, jj, pp, rr, cnt + 1)
                seen.remove((ii, jj, pp, rr))

    dfs(*red, *blue, cnt=0)
    return ans if ans != INF else -1

print(solution1(deepcopy(s), verbose=True))
```

</div>

{:.output_stream}

```
WorkingTime[solution1]: 3.67522 ms
7

```

## BFS Approach

For this problem, BFS search is more easy to think. <br>
This is because greed searching is possible. 
Recall that our objective is finding shortest path <br>
from red ball `'R'` to a hole `'O'` by considering the blue ball `'B'`. <br>
BFS can search the hole by increasing counting of the trial(a behavior of sloping the table).

<div class="prompt input_prompt">
In&nbsp;[137]:
</div>

<div class="input_area" markdown="1">

```python
up = left = -1
down = right = 1
INF = 1e20

@logging_time
def solution2(s):
    n, m = len(s), len(s[0])
    red = [-1, -1]
    blue = [-1, -1]
    for i in range(n):
        for j in range(m):
            if s[i][j] == 'R':
                red = [i, j]
                s[i][j] = '.'
            elif s[i][j] == 'B':
                blue = [i, j]
                s[i][j] = '.'

    def go(i, j, row, col):
        x, y = i, j
        walk = 0
        while s[x + row][y + col] == '.':
            x, y = x + row, y + col
            walk += 1
        if s[x + row][y + col] == 'O':
            x, y = x + row, y + col
            walk += 1
        return (x, y), walk

    def adj(i, j, p, r):
        for x, y, z in [(up, 0, 'up'), (down, 0, 'down'), (0, right, 'right'), (0, left, 'left')]:
            nextred, walkred = go(i, j, row=x, col=y)
            nextblue, walkblue = go(p, r, row=x, col=y)
            if nextred == nextblue and s[nextred[0]][nextred[1]] != 'O':
                if walkred < walkblue:
                    nextblue = (nextblue[0] - x, nextblue[1] - y)
                else:
                    nextred = (nextred[0] - x, nextred[1] - y)
            if (i, j, p, r) != (*nextred, *nextblue):
                yield (*nextred, *nextblue, z)

    def bfs(i, j, p, r):
        ans = INF
        seen = set()
        queue = deque([(i, j, p, r, 0)])
        seen.add((i, j, p, r, 0))
        while queue:
            i, j, p, r, cnt = queue.popleft()

            if cnt > 10 or s[p][r] == 'O':
                continue
                
            if s[i][j] == 'O':
                ans = min(ans, cnt)
                break

            for ii, jj, pp, rr, z in adj(i, j, p, r):
                if (ii, jj, pp, rr, z) not in seen:
                    seen.add((ii, jj, pp, rr))
                    queue.append((ii, jj, pp, rr, cnt + 1))
        return ans if ans != INF else -1
    return bfs(*red, *blue)

print(solution2(deepcopy(s), verbose=True))
```

</div>

{:.output_stream}

```
WorkingTime[solution2]: 2.83623 ms
7

```

## Test(generate toy examples)

<div class="prompt input_prompt">
In&nbsp;[138]:
</div>

<div class="input_area" markdown="1">

```python
n, m = 10, 8
s = [['#'] * m for _ in range(n)]
for _ in range(1, n - 1):
    for _ in range(1, m - 1):
        s[random.randint(1, n - 2) ][random.randint(1, m - 2) ] = '.'
s[random.randint(1, n - 2) ][random.randint(1, m - 2)] = 'B'
s[random.randint(1, n - 2) ][random.randint(1, m - 2)] = 'R'
s[random.randint(1, n - 2) ][random.randint(1, m - 2)] = 'O'
print(np.array(s))

print(solution1(deepcopy(s), verbose=True))
print(solution2(deepcopy(s), verbose=True))
```

</div>

{:.output_stream}

```
[['#' '#' '#' '#' '#' '#' '#' '#']
 ['#' '.' '.' '.' '#' '.' '.' '#']
 ['#' '.' '.' '#' '#' '.' '.' '#']
 ['#' '.' '.' '.' '#' '.' '.' '#']
 ['#' '#' '.' '.' '.' '#' '.' '#']
 ['#' '#' '#' 'O' '#' 'B' '.' '#']
 ['#' '#' '.' '.' '#' '.' '#' '#']
 ['#' '#' '.' '.' '.' '.' 'R' '#']
 ['#' '#' '.' '.' '.' '.' '.' '#']
 ['#' '#' '#' '#' '#' '#' '#' '#']]
WorkingTime[solution1]: 10.98490 ms
3
WorkingTime[solution2]: 0.11563 ms
3

```

<div class="prompt input_prompt">
In&nbsp;[139]:
</div>

<div class="input_area" markdown="1">

```python
%timeit -r 2 -n 100 solution1(deepcopy(s), verbose=False)
%timeit -r 2 -n 100 solution2(deepcopy(s), verbose=False)
```

</div>

{:.output_stream}

```
8.9 ms ± 17.8 µs per loop (mean ± std. dev. of 2 runs, 100 loops each)
148 µs ± 1 µs per loop (mean ± std. dev. of 2 runs, 100 loops each)

```

<div class="prompt input_prompt">
In&nbsp;[140]:
</div>

<div class="input_area" markdown="1">

```python
reconstructed = ''
for ss in s:
    reconstructed += (''.join(ss) + '\n')
print(reconstructed)
```

</div>

{:.output_stream}

```
########
#...#..#
#..##..#
#...#..#
##...#.#
###O#B.#
##..#.##
##....R#
##.....#
########


```

## Summited Code 

```python
from sys import stdin
from collections import deque
import sys
sys.setrecursionlimit(10000)
up = left = -1
down = right = 1
INF = 1e20

def solution(s):
    # TODO

stdin = open("./data/Escape.txt")  # 백준에 제출할 시에는 주석 처리                
n, m = map(int, stdin.readline().split(' '))
s = [list(stdin.readline().strip()) for _ in range(n)]
print(solution(s))
```
