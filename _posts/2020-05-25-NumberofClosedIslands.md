---
title: "1254.Number of Closed Island"
excerpt: "2 step of DFS, union find practice"
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
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
import sys, random
sys.path.append("/home/swyoo/algorithm/")
from typing import List
import numpy as np
from utils.verbose import visualize_ds, logging_time
```

</div>

# 1254. Number of Closed Islands

Let $m, n$ be grid shape. 

## DFS + Disjoint Set(or Union Find)

### Idea
1. Remove lands connected to edges by re-marking grid elements.
2. Union all grid except for water.
3. Count all representatives.

The time complexity is $O(mn\alpha)$

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
class Solution1:
    @logging_time
    def closedIsland(self, grid: List[List[int]], show=False) -> int:
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

        def adj(i, j):
            for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and not grid[x][y]:
                    yield x, y

        seen = set()
        def rmland(i, j):
            seen.add((i, j))
            grid[i][j] = 1
            for x, y in adj(i, j):
                if (x, y) not in seen:
                    rmland(x, y)

        for i in range(len(grid)):
            if (i, 0) not in seen and not grid[i][0]:
                rmland(i, 0)
            if (i, len(grid[0]) - 1) not in seen and not grid[i][len(grid[0]) - 1]:
                rmland(i, len(grid[0]) - 1)

        for j in range(len(grid[0])):
            if (0, j) not in seen and not grid[0][j]:
                rmland(0, j)
            if (len(grid) - 1, j) not in seen and not grid[len(grid) - 1][j]:
                rmland(len(grid) - 1, j)

        if show: print(np.array(grid))

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if not grid[i][j]:
                    find((i, j))
                    for x, y in adj(i, j):
                        union((i, j), (x, y))
        if show: visualize_ds(par)

        ans = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if not grid[i][j] and (i, j) == find((i,j)):
                    ans += 1
        return ans
    
sol1 = Solution1()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
grid = [[1,1,1,1,1,1,1,0],[1,0,0,0,0,1,1,0],[1,0,1,0,1,1,1,0],[1,0,0,0,0,1,0,1],[1,1,1,1,1,1,1,0]]
sol1.closedIsland(grid, show=True, verbose=True)
```

</div>

{:.output_stream}

```
[[1 1 1 1 1 1 1 1]
 [1 0 0 0 0 1 1 1]
 [1 0 1 0 1 1 1 1]
 [1 0 0 0 0 1 0 1]
 [1 1 1 1 1 1 1 1]]

```


![png](/assets/images/NumberofClosedIslands_files/NumberofClosedIslands_4_1.png)


{:.output_stream}

```
WorkingTime[closedIsland]: 191.04004 ms

```




{:.output_data_text}

```
2
```



## DFS

### Idea
`seen` is very important!.
1. Remove lands connected to edges by re-marking grid elements and add all grid to `seen`.
2. add all lands to `seen`.

notice that once `dfs(i, j)` marks all lands connected to `(i, j)` except for waters.

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
class Solution2:
    @logging_time
    def closedIsland(self, grid: List[List[int]], show=False) -> int:

        def adj(i, j):
            for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and not grid[x][y]:
                    yield x, y

        seen = set()
        def dfs(i, j, rmland=False):
            seen.add((i, j))
            if rmland: grid[i][j] = 1
            for x, y in adj(i, j):
                if (x, y) not in seen:
                    dfs(x, y)

        for i in range(len(grid)):
            if (i, 0) not in seen and not grid[i][0]:
                dfs(i, 0, rmland=True)
            if (i, len(grid[0]) - 1) not in seen and not grid[i][len(grid[0]) - 1]:
                dfs(i, len(grid[0]) - 1, rmland=True)

        for j in range(len(grid[0])):
            if (0, j) not in seen and not grid[0][j]:
                dfs(0, j, rmland=True)
            if (len(grid) - 1, j) not in seen and not grid[len(grid) - 1][j]:
                dfs(len(grid) - 1, j, rmland=True)

        if show: print(np.array(grid))

        ans = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if (i, j) not in seen and not grid[i][j]:
                    ans += 1
                    dfs(i, j)
        return ans

sol2 = Solution2()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[5]:
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
[[1 1 0 ... 1 0 0]
 [1 0 1 ... 1 1 0]
 [1 0 0 ... 0 1 1]
 ...
 [0 0 0 ... 0 1 1]
 [0 0 1 ... 1 0 0]
 [0 1 0 ... 1 0 1]]

```

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
ans1 = sol1.closedIsland(grid, verbose=True)
ans2 = sol2.closedIsland(grid, verbose=True)
print(ans1, ans2)
assert ans1 == ans2
```

</div>

{:.output_stream}

```
WorkingTime[closedIsland]: 2841.18605 ms
WorkingTime[closedIsland]: 1092.00549 ms
64681 64681

```
