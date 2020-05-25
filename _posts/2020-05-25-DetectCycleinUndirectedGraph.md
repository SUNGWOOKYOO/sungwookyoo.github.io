---
title: "Detect Cycle in an Undirected Graph"
excerpt: "using disjoint set, easily check if a graph has any cycle."
categories:
 - algorithms
tags:
 - datastructure
 - union find
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
In&nbsp;[9]:
</div>

<div class="input_area" markdown="1">

```python
import sys
sys.path.append("/home/swyoo/algorithm/")
from utils.verbose import visualize_ds
```

</div>

# Detect Cycles in an Undirected Graph

<div class="prompt input_prompt">
In&nbsp;[11]:
</div>

<div class="input_area" markdown="1">

```python
def solution(graph):

    par, rnk = {}, {}

    def find(x):
        if x not in par:
            par[x] = x
            rnk[x] = 0
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

    # enumerate all edges
    for u, vlist in enumerate(graph):
        for v in vlist:
            x, y = find(u), find(v)
            if x == y:
                print("detect cycle: {} with {}".format(u, v))
                return par
            else:
                union(x, y)
    return par
```

</div>

<div class="prompt input_prompt">
In&nbsp;[12]:
</div>

<div class="input_area" markdown="1">

```python
graph = [[1,2,3],[0,2],[0,1],[0,4],[3]]
visualize_ds(solution(graph))
```

</div>

{:.output_stream}

```
detect cycle: 1 with 0

```


![png](/assets/images/DetectCycleinUndirectedGraph_files/DetectCycleinUndirectedGraph_3_1.png)

