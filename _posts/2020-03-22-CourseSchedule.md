---
title: "207.Course Schedule"
excerpt: "KEY: using BFS or DFS, detect cycles in a graph"
categories:
 - algorithms
tags:
 - BFS
 - DFS
use_math: true
last_modified_at: "2020-03-22"
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
    url: "https://leetcode.com/problems/course-schedule/"
---

# 207. Course Schedule

[leetcode](https://leetcode.com/problems/course-schedule/)

## Key Idea
Create graph from given `prerequisites`, and then check if cycles exist in the graph. <br>
If any cycle exist, taking all courses is impossible. <br>

Using topological sort by DFS or BFS, checking cycles is possible. See [this post](https://sungwookyoo.github.io/algorithms/TopologicalSort/)


## Use DFS

<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
from collections import defaultdict

class Solution(object):
    def __init__(self):
        self.isDAG = True

    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        adj = defaultdict(list)
        for i, j in prerequisites:
            adj[j].append(i)

        seen = [False] * numCourses
        finish = [False] * numCourses
        topo = []

        def dfs(i):
            seen[i] = True
            for j in adj[i]:
                if not seen[j]:
                    dfs(j)
                elif not finish[j]:
                    self.isDAG = False
            finish[i] = True
            topo.append(i)

        for i in range(numCourses):
            if not seen[i]:
                dfs(i)
        if not self.isDAG:
            return False
        else:
            return len(topo) == numCourses
```

</div>

{:.output_stream}

```
True

```

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
numCourses = 2
prerequisites = [[0, 1]]
sol = Solution()
print(sol.canFinish(numCourses, prerequisites))
```

</div>

{:.output_stream}

```
True

```

## Use BFS

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
from collections import defaultdict, deque

class Solution(object):
    def __init__(self):
        self.isDAG = True

    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        adj = defaultdict(list)
        indegree = [0] * numCourses
        for i, j in prerequisites:
            adj[j].append(i)
            indegree[i] += 1
        
        st = [i for i in range(numCourses) if indegree[i] == 0]
        queue = deque(st)
        cnt = 0
        while queue:
            i = queue.popleft()
            for j in adj[i]:
                indegree[j] -= 1
                if indegree[j] == 0:
                    queue.append(j)
            cnt += 1
        return cnt == numCourses
```

</div>

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
numCourses = 2
prerequisites = [[0, 1]]
sol = Solution()
print(sol.canFinish(numCourses, prerequisites))
```

</div>

{:.output_stream}

```
True

```
