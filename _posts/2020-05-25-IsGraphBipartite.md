---
title: "785.Is Graph Bipartite"
excerpt: "use DFS or BFS, check if a graph is bipartite or not."
categories:
 - algorithms
tags:
 - DFS
 - BFS
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
In&nbsp;[28]:
</div>

<div class="input_area" markdown="1">

```python
import sys
sys.path.append("/home/swyoo/algorithm/")
from utils.verbose import visualize_graph
from typing import List
from collections import defaultdict, deque
```

</div>

# 785. Is Graph Bipartite?

## Idea
If a graph is bipartite, two coloring is possible! <br>
If conflict is occured while coloring process, isBipartite flag become `False`. <br>
I saw this idea from [this document](https://iq.opengenus.org/bipartite-checking-bfs/)

## DFS

<div class="prompt input_prompt">
In&nbsp;[30]:
</div>

<div class="input_area" markdown="1">

```python
class Solution1:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        color = {}  # True: red, False: black
        isbipartite = True
        def dfs(i, paint):
            nonlocal isbipartite
            color[i] = paint
            for j in graph[i]:
                if j not in color:
                    dfs(j, paint=~color[i])
                elif color[i] == color[j]:
                    isbipartite = False
        for x in range(len(graph)):
            if x not in color:
                dfs(x, paint=False)
        return isbipartite
    
sol1 = Solution1()    
```

</div>

<div class="prompt input_prompt">
In&nbsp;[31]:
</div>

<div class="input_area" markdown="1">

```python
# toy example
edges = [[0,1],[1,2],[2,3],[2,4],[3,5],[4,5],[3,6],[4,8],[6,7],[5,9],[9,10],[10,11],[11,12],[10,12]]
graph = defaultdict(list)
for u,v in edges:
    graph[u].append(v)
visualize_graph(edges, undirected=True)
```

</div>


![png](/assets/images/IsGraphBipartite_files/IsGraphBipartite_4_0.png)


<div class="prompt input_prompt">
In&nbsp;[32]:
</div>

<div class="input_area" markdown="1">

```python
sol.isBipartite(graph)
```

</div>




{:.output_data_text}

```
False
```



<div class="prompt input_prompt">
In&nbsp;[33]:
</div>

<div class="input_area" markdown="1">

```python
class Solution2:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        color = {}  # True: red, False: black
        isBipartite = True
        def bfs(s):
            nonlocal isBipartite
            color[s] = False
            queue = deque([(s, color[s])])
            while queue:
                i, paint = queue.popleft()
                for j in graph[i]:
                    if j not in color:
                        color[j] = ~paint
                        queue.append((j, color[j]))
                    elif paint == color[j]:
                        isBipartite = False
        for x in range(len(graph)):
            if x not in color:
                bfs(x)
        return isBipartite
sol2 = Solution2()    
```

</div>

<div class="prompt input_prompt">
In&nbsp;[34]:
</div>

<div class="input_area" markdown="1">

```python
sol2.isBipartite(graph)
```

</div>




{:.output_data_text}

```
False
```



# Reference
[1] [a detailed document](https://iq.opengenus.org/bipartite-checking-bfs/) <br>
[2] [leetcode problem](https://leetcode.com/problems/is-graph-bipartite/submissions/)
