---
title: "695.Max Area of Island"
excerpt: "Given many islands, we want to find maximum size of island."

categories:
  - algorithms
tags:
  - DFS
use_math: true
last_modified_at: 2020-
toc: true
toc_sticky: true
toc_label: ""
toc_icon: "cog"
---

[leetcode](https://leetcode.com/problems/max-area-of-island/)

  using DFS, I solved this problem.  
I implemented DFS recursive style, and iterative style to practice DFS.

## Key idea
  In this grid wolrd, check each grid and count the size of islands by DFS.  
At first, `1` can be a start point of DFS, note that if visited once, this grid cannot be visited again.  
Whenever exploring a new grid, size will be updated by +1.
This algorithm can be implemented by recursively or iteratively.

### Recursive Approach
  Note that `loc` value will be the size of an island.  
It is hard to think `loc = 1` when starting a `DFS` function.  
This is because it is a top-down manner.  
To be more specific,  
a grid make `1` at the beginning of a function and aggregate all grids in the for loop.  
Finally, the `DFS` function returns the aggregated result at the finishing time.

```python
import numpy as np

class Solution(object):
    def maxAreaOfIsland(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        seen = set()
        grid = np.array(grid)
        m, n = grid.shape
        
        # recursive 
        def adj(i, j):
            for x, y in [(i-1, j),(i+1, j),(i, j-1),(i, j+1)]:
                if 0 <= x < m and 0 <= y < n:
                    yield x, y
                    
        def DFS(i, j):
            seen.add((i, j))
            loc = 1 # loc value will be the size of an island.
            for x, y in adj(i, j):
                if (x, y) not in seen and (grid[x, y] == 1):
                    loc += DFS(x, y) # aggregatation.
            return loc # finishing time.
        
        sol_recursive = 0
        for i in range(m):
            for j in range(n):
                if grid[i, j] == 1 and (i, j) not in seen:
                    sol_recursive = max(sol_recursive, DFS(i, j)) 
        seen.clear()
        return sol_recursive
```

### Iterative Approach
 This approach is easy to understand than above case.

```python
# iteration
def DFS_iter():
    ans = 0
    for i in range(m):
        for j in range(n):
            if grid[i, j] == 1 and (i, j) not in seen:
            	local = DFS_iteration(i,j)
            	ans = max(ans, local)
 	return ans
```

```python
def DFS_iteration(i, j):
    local = 0
    stack = [(i, j)]
    seen.add((i, j))
    while stack:
        r, c = stack.pop()
        local += 1
        for x, y in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if 0 <= x < m and 0 <= y < n and (x, y) not in seen and grid[x, y] == 1:
                stack.append((x, y))
                seen.add((x, y))
    return local
```



## Report

recursive방식으로 DFS를 사용할때, 함수의 시작부에서 `loc` 값을 `1` 로 해야한다는 점이 약간 까다로웠다. 
{: .notice-warning}


