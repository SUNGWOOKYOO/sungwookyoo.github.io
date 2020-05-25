---
title: "200.Number of Islands"
excerpt: "practice of DFS and union find"
categories:
 - algorithms
tags:
 - datastructure
 - union find
 - DFS
use_math: true
last_modified_at: "2020-05-25"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/teaser.jpg
 overlay_filter: 0.5
---

<div class="prompt input_prompt">
In&nbsp;[19]:
</div>

<div class="input_area" markdown="1">

```python
import sys, random
sys.path.append("/home/swyoo/algorithm/")
from utils.verbose import visualize_graph, visualize_ds, logging_time
from typing import List
from collections import deque
from pprint import pprint
import numpy as np
```

</div>

# 200. Number of Islands
[leetcode](https://leetcode.com/problems/number-of-islands/submissions/)

## Disjoint set

1. Union adjacent and reachable points by iterating all entries. 
2. Count representatives. 

<div class="prompt input_prompt">
In&nbsp;[12]:
</div>

<div class="input_area" markdown="1">

```python
class Solution1:
    @logging_time
    def numIslands(self, grid: List[List[str]], show=False) -> int:
        par, rnk = {}, {}

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
        
        if show: print(np.array(grid))
        def adj(i, j):
            for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and int(grid[x][y]):
                    yield x, y

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if int(grid[i][j]):
                    find((i, j))
                    for x, y in adj(i, j):
                        union((i, j), (x, y))
        if show: visualize_ds(par)
        ans = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if int(grid[i][j]) and (i,j) == find((i, j)):
                    ans += 1
        return ans

sol1 = Solution1()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[13]:
</div>

<div class="input_area" markdown="1">

```python
# grid = [["1", "1", "1", "1", "0"], ["1", "1", "0", "1", "0"], ["1", "1", "0", "0", "0"], ["0", "0", "0", "0", "0"]]
grid = \
[[1,1,0,0,0],
[1,1,0,0,0],
[0,0,1,0,0],
[0,0,0,1,1]]
print("ans:", sol1.numIslands(grid, verbose=True, show=True))
```

</div>

{:.output_stream}

```
[[1 1 0 0 0]
 [1 1 0 0 0]
 [0 0 1 0 0]
 [0 0 0 1 1]]

```


![png](/assets/images/NumberofIslands_files/NumberofIslands_4_1.png)


{:.output_stream}

```
WorkingTime[numIslands]: 172.88089 ms
ans: 3

```

## DFS 

1. Mark adjacent reachable grid as seen from start point. 
2. Count the number of start points. 

<div class="prompt input_prompt">
In&nbsp;[14]:
</div>

<div class="input_area" markdown="1">

```python
class Solution2:
    @logging_time 
    def numIslands(self, grid: List[List[str]]) -> int:

        def adj(i, j):
            for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and int(grid[x][y]):
                    yield x, y

        seen = set()
        def dfs(i, j):
            seen.add((i, j))
            # grid[i][j] = '2'
            for x, y in adj(i, j):
                if (x, y) not in seen:
                    dfs(x, y)

        ans = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if int(grid[i][j]) and (i, j) not in seen:
                    ans += 1
                    dfs(i, j)
        return ans
sol2 = Solution2()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[18]:
</div>

<div class="input_area" markdown="1">

```python
grid = \
[[1,1,0,0,0],
[1,1,0,0,0],
[0,0,1,0,0],
[0,0,0,1,1]]
print("ans1:", sol1.numIslands(grid, verbose=True, show=False))
print("ans2:", sol2.numIslands(grid, verbose=True))
```

</div>

{:.output_stream}

```
WorkingTime[numIslands]: 0.05245 ms
ans1: 3
WorkingTime[numIslands]: 0.02980 ms
ans2: 3

```

<div class="prompt input_prompt">
In&nbsp;[32]:
</div>

<div class="input_area" markdown="1">

```python
m, n = 1000, 1000
grid = [[random.randint(0, 1) for i in range(m)] for j in range(n)] 
print(np.array(grid))
```

</div>

{:.output_stream}

```
[[0 1 0 ... 1 1 0]
 [0 1 1 ... 1 0 0]
 [1 1 1 ... 0 0 1]
 ...
 [0 1 0 ... 0 0 1]
 [0 0 0 ... 1 1 1]
 [0 0 0 ... 1 0 1]]

```

<div class="prompt input_prompt">
In&nbsp;[33]:
</div>

<div class="input_area" markdown="1">

```python
print("ans1:", sol1.numIslands(grid, verbose=True, show=False))
print("ans2:", sol2.numIslands(grid, verbose=True))
```

</div>

{:.output_stream}

```
WorkingTime[numIslands]: 3131.24204 ms
ans1: 66540
WorkingTime[numIslands]: 1263.87072 ms
ans2: 66540

```
