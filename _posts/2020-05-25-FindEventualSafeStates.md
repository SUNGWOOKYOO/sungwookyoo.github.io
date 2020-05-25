---
title: "802.Find Eventual Safe States"
excerpt: "practice of DFS, this is an application of detecting cycle."
categories:
 - algorithms
tags:
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
from utils.verbose import logging_time
from typing import List
import numpy as np
sys.setrecursionlimit(100000)
```

</div>

# Find Eventual Safe States

Which nodes are eventually safe?  Return them as an array in sorted order.
[detail](https://leetcode.com/problems/find-eventual-safe-states/)

## Naive DFS 
$$
O((V + E)^2)
$$

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
class Solution1:
    def __init__(self):
        self.isCycle = False
    @logging_time
    def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
        def dfs(i):
            """ detect cycle. """
            seen[i] = True
            for j in graph[i]:
                if j not in seen:
                    dfs(j)
                elif j not in finish:
                    # print("detect cycle")
                    self.isCycle = True
                    return
            finish[i] = True

        ans = []
        for x in range(len(graph)):
            seen, finish = {}, {}
            self.isCycle = False
            dfs(x)
            if not self.isCycle:
                ans.append(x)
        return ans

sol1 = Solution1()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
# graph = [[1,2],[2,3],[5],[0],[5],[],[]]
graph = [[1,2,3,4],[1,2],[3,4],[0,4],[]]
print(sol1.eventualSafeNodes(graph, verbose=True))
```

</div>

{:.output_stream}

```
WorkingTime[eventualSafeNodes]: 0.00978 ms
[4]

```

## Improved DFS 
I was inspired by a [Youtube](https://youtu.be/6ySoJbyBs4E) video. <br>
The time complexity will be as follows.
$$
O((V + E))
$$

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
class Solution2:
    @logging_time
    def eventualSafeNodes(self, graph):
        seen, finish, ans = set(), set(), set()
        def dfs(i):
            seen.add(i)
            for j in graph[i]:
                if j not in seen:
                    if dfs(j):
                        return True
                elif j not in finish:
                    # print("({} -> {}) edge makes cycle".format(i, j))
                    return True
            finish.add(i)
            ans.add(i)
            return False
        for x in range(len(graph)):
            if x not in seen:
                dfs(x)
        return sorted(list(ans))
sol2 = Solution2()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
# graph = [[1,2,3,4],[1,2],[3,4],[0,4],[]]
sol2.eventualSafeNodes(graph, verbose=True)
```

</div>

{:.output_stream}

```
WorkingTime[eventualSafeNodes]: 0.00763 ms

```




{:.output_data_text}

```
[4]
```



Note:

* `graph` will have length at most `10000`.
* The number of edges in the `graph` will not exceed `32000`.
* Each `graph[i]` will be a sorted list of different integers, chosen within the range `[0, graph.length - 1]`.

<div class="prompt input_prompt">
In&nbsp;[7]:
</div>

<div class="input_area" markdown="1">

```python
size, rate = 5000, 0.1
graph = []
for x in range(size):
    graph.append(list(np.random.choice(size, int(random.randint(0, size - 1) * rate), replace=False)))
    
print(sol1.eventualSafeNodes(graph, verbose=True))
print(sol2.eventualSafeNodes(graph, verbose=True))
```

</div>

{:.output_stream}

```
WorkingTime[eventualSafeNodes]: 15085.39867 ms
[109, 377, 775, 1174, 1943, 2577, 3140, 3748, 4251]
WorkingTime[eventualSafeNodes]: 2.45190 ms
[109, 377, 775, 1174, 1943, 2577, 3140, 3748, 4251]

```

# Summary 
This problem is a similar problem with detect cycle in a directed graph. <br>
I refered some pages. <br>
If you want to know about more, please visit this documents as follows.

# Reference

[1] [Detect Cycle using topolgoical sort python implementation](https://sungwookyoo.github.io/algorithms/TopologicalSort/)

<div class="prompt input_prompt">
In&nbsp;[None]:
</div>

<div class="input_area" markdown="1">

```python

```

</div>
