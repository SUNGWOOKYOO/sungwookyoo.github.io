---
title: "417.Pacific Atlantic Water Flow"
excerpt: "Check if there are ways to go both Pacific, Atlantic ocean "
categories:
 - algorithms
tags:
 - DFS
 - BFS
use_math: true
last_modified_at: "2020-03-24"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
overlay_image: /assets/images/algorithms/algo.png
overlay_filter: 0.5
 caption: algorithms
 actions:
  - label: "leetcode"
    url: "https://leetcode.com/problems/pacific-atlantic-water-flow"
---

<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
import numpy as np
import sys
sys.path.append("/home/swyoo/algorithm")
from utils.verbose import logging_time
```

</div>

# 417. Pacific Atlantic Water Flow

## First Approach

DFS를 사용해서 `(i,j)` 에서 시작해서 동,서,남,북 중 가능한 경로로 모두 가보자. <br>
그러다 pacific ocean이나 Atlantic ocean이 나오면(**끝까지 가봐야 알 수있음**) flag를 `True`로 만들자. <br>
모든 경로를 가보기 위해서 revert과정이 필요. <br>
시간복잡도는 `(i,j)` 하나의 entry당 $O(4^{mn})$ 만큼 탐색을 해봐야한다. <br>
따라서, $O(mn4^{mn})$ (height가 같거나 낮은 곳으로만 water가 흐를 수 있기 때문에 pruning이 많이 된다).

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
class Solution(object):
    @logging_time
    def pacificAtlantic(self, matrix, visual=False):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
        if matrix == []: return []
        m, n = len(matrix), len(matrix[0])

        def adj(i, j):
            for x, y in [(i, j + 1), (i, j - 1), (i - 1, j), (i + 1, j)]:
                if (-1 <= x <= m) and (-1 <= y <= n):
                    yield x, y

        seen = [[False] * n for _ in range(m)]
        check = [False, False]  # pacific flag, atlantic flag

        # visual = lambda x: print(np.array(x, dtype=int))

        def visualize(seen):
            x = np.array(matrix, dtype=str)
            for i in range(m):
                for j in range(n):
                    if seen[i][j]:
                        x[i, j] = '#'
            print(x)

        def dfs(i, j, visual=False):
            seen[i][j] = True
            if visual: visualize(seen)
            for x, y in adj(i, j):
                if x == -1 or y == -1:
                    check[0] = True
                elif x == m or y == n:
                    check[1] = True

                if check == [True, True]:
                    return

                if (0 <= x < m) and (0 <= y < n) and (not seen[x][y]):
                    if matrix[x][y] <= matrix[i][j]:
                        dfs(x, y)
            # revert
            seen[i][j] = False

        ans = []
        # dfs(0, 4)
        # print(check)

        for i in range(m):
            for j in range(n):
                seen = [[False] * n for _ in range(m)]
                check = [False, False]
                dfs(i, j, visual=visual)
                if check == [True, True]:
                    ans.append([i, j])
        return ans
```

</div>

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
sol = Solution()
x = [[1, 2, 2, 3, 5], [3, 2, 3, 4, 4], [2, 4, 5, 3, 1], [6, 7, 1, 4, 5], [5, 1, 1, 2, 4]]
print(sol.pacificAtlantic(x, visual=False, verbose=True))
```

</div>

{:.output_stream}

```
WorkingTime[pacificAtlantic]: 0.85282 ms
[[0, 4], [1, 3], [1, 4], [2, 2], [3, 0], [3, 1], [4, 0]]

```

결과는 113 / 113 test cases passed, **but took too long**. <br>
더 효율적인 방식의 접근이 필요하다.

## Better Approach 

[good code](https://leetcode.com/problems/pacific-atlantic-water-flow/discuss/438276/Python-beats-98.-DFS-template-for-Matrix) 에서 좋은 방식을 찾았다. <br>
관점을 바꿔서, 가장자리에서 dfs를 시작한다면, <br>
이미 pacific, 또는 atlantic ocean에 도달할 수있는 것을 알고 시작하는 것이다. <br>
따라서, reachable한 지점들을 모아 intersection하면, 답이 된다.

시간복잡도는 **각 entry별로 reachable한지 <br>
pacific, atlantic에 대해 각각 한번씩만 check하면** <br>
곧바로 답이 될 수 있는지 알 수 있다. <br>
따라서, $O(mn)$ 이된다.

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
import numpy as np
from collections import defaultdict


class Solution(object):
    @logging_time
    def pacificAtlantic(self, matrix, visual=False):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
        if matrix == []: return []
        m, n = len(matrix), len(matrix[0])

        def adj(i, j):
            for x, y in [(i, j + 1), (i, j - 1), (i - 1, j), (i + 1, j)]:
                if (0 <= x < m) and (0 <= y < n):
                    yield x, y

        pacific = set()
        atlantic = set()

        def visualize(seen, fill='#'):
            x = np.array(matrix, dtype=str)
            for i in range(m):
                for j in range(n):
                    if (i, j) in seen:
                        x[i, j] = fill
            print(x)

        def dfs(i, j, seen: set, visual=False, fill='#'):
            """ search """
            seen.add((i, j))
            if visual: visualize(seen, fill)
            for x, y in adj(i, j):
                if (x, y) not in seen and (matrix[x][y] >= matrix[i][j]):
                    dfs(x, y, seen)

        # update pacific, atlantic
        for i in range(m):
            dfs(i, 0, seen=pacific, visual=visual, fill='~')
            dfs(i, n - 1, seen=atlantic, visual=visual, fill='*')
        for j in range(n):
            dfs(0, j, seen=pacific, visual=visual, fill='~')
            dfs(m - 1, j, seen=atlantic, visual=visual, fill='*')
        return list(pacific & atlantic)
```

</div>

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
sol = Solution()
x = [[1, 2, 2, 3, 5], [3, 2, 3, 4, 4], [2, 4, 5, 3, 1], [6, 7, 1, 4, 5], [5, 1, 1, 2, 4]]
print(sol.pacificAtlantic(x, visual=False, verbose=True))
```

</div>

{:.output_stream}

```
WorkingTime[pacificAtlantic]: 0.15402 ms
[(1, 3), (3, 0), (3, 1), (1, 4), (0, 4), (2, 2), (4, 0)]

```
