---
title: "329.Longest Increasing Path - Leetcode"
excerpt: "#"
categories:
 - algorithms
tags:
 - DFS
use_math: true
last_modified_at: "2020-03-24"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/algorithms/algo.png
 overlay_filter: 0.5
 caption: algorithm
 actions:
  - label: "leetcode"
    url: "https://leetcode.com/problems/longest-increasing-path-in-a-matrix/"
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

# 329. Longest Increasing Path in a Matrix

모든 case 를 보는 것은 $O(mn4^{mn})$이기 때문에 매우 비효율적이다.

## Key Idea
**`memoization`과 함께 `dfs`를 사용하자.** <br>

따라서, 다음과 같이 구현하면 된다. <br>
<span style="color:red">`dfs(i,j)`를 한번 call하면 `(i,j)`에서 시작하여 거치는 모든 entry들에 대해 시작하여 끝마칠때까지의 최대 거리가 memo된다</span>.

`dfs(i, j)`는 `(i,j)` 에서 시작하여 갈 수있는 최대의 path 길이를 구한다. <br>
만약 `dfs(r,c)` 를 call했을때, `(i,j)`를 거치게 되어 `dfs(i,j)`를 call하면, <br>
이전에 `dfs(i,j)`가 call되었었다는 가정하에 <br>
구해 놓은 `(i,j)`를 지나는 최대의 path를 저장해 놓은 값을 그대로 사용하기 때문에 <br>
`dfs(i, j)`를 또 다시 recursion 할 필요가 없어 시간이 매우 절약된다. <br>

모든 entry $mn$개에 대해 recursion은 한번만 발생하고 동,서,남,북을 뒤지기만 하면 되므로 <br>
하나의 entry를 채우는 시간은 $O(1)$이다. <br>
따라서, 시간복잡도는 $O(mn)$ 이다.

### 주의사항
`res`값과 `loc`을 따로 두고, res와 loc중 max값을 저장해야함에 유의하자. <br>
왜냐하면, 어떤 하나의 entry에서 시작하는 최장거리의 path길이를 구할할때 <br>
이웃한 하나의 entry(예를 들면 동쪽)가 finish 되더라도, <br>
또 다른 이웃한 entry(예를 들면 서쪽)에 의해 더 장거리를 갈 수있는 path를 발견할 수 있기 때문이다.
![](/assets/images/algorithms/LongestPath.PNG){:width="300"}

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
class Solution(object):
    @logging_time
    def longestIncreasingPath(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: int
        """
        if matrix == []: return 0
        m, n = len(matrix), len(matrix[0])

        def adj(i, j):
            for x, y in [(i, j + 1), (i, j - 1), (i - 1, j), (i + 1, j)]:
                if (0 <= x < m) and (0 <= y < n):
                    yield x, y

        memo = dict()

        def dfs(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            res, loc = 1, 1
            for x, y in adj(i, j):
                if matrix[x][y] > matrix[i][j]:
                    loc = 1 + dfs(x, y)
                # take max value from all recursion passing matrix[i][j]
                res = max(res, loc)
            memo[(i, j)] = res
            return res

        ans = 1
        for i in range(m):
            for j in range(n):
                ans = max(ans, dfs(i, j))
        return ans


sol = Solution()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
x = [[0,1,2,3,4,5,6,7,8,9],[19,18,17,16,15,14,13,12,11,10],[20,21,22,23,24,25,26,27,28,29],[39,38,37,36,35,34,33,32,31,30],[40,41,42,43,44,45,46,47,48,49],[59,58,57,56,55,54,53,52,51,50],[60,61,62,63,64,65,66,67,68,69],[79,78,77,76,75,74,73,72,71,70],[80,81,82,83,84,85,86,87,88,89],[99,98,97,96,95,94,93,92,91,90],[100,101,102,103,104,105,106,107,108,109],[119,118,117,116,115,114,113,112,111,110],[120,121,122,123,124,125,126,127,128,129],[139,138,137,136,135,134,133,132,131,130],[0,0,0,0,0,0,0,0,0,0]]
print(sol.longestIncreasingPath(x, verbose=True))
```

</div>

{:.output_stream}

```
WorkingTime[longestIncreasingPath]: 0.47326 ms
140

```

Reference [[2]](https://leetcode.com/problems/pacific-atlantic-water-flow/)와 비슷한 문제이다.

# Reference
[1] [good code and explanation](https://leetcode.com/problems/pacific-atlantic-water-flow/discuss/438276/Python-beats-98.-DFS-template-for-Matrix) <br>
[2] [417. Pacific Atlantic Water Flow](https://leetcode.com/problems/pacific-atlantic-water-flow/)
