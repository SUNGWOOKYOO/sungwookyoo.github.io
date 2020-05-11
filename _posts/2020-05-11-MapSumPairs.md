---
title: "677.MapSumPairs"
excerpt: "Trie datastructure practice"
categories:
 - algorithms
tags:
 - datastructure
use_math: true
last_modified_at: "2020-05-11"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/algorithms/algo.png
 overlay_filter: 0.5
---

<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
from collections import defaultdict
from pprint import pprint
```

</div>

# 677. Map Sum Pairs

## Idea
1. Trie datastructure is used to store key, value. 
2. `sum` function is operated as follows. 
* The algorithm goes into the final depth of the given prefix, I will call this node in the depth as `cur`.
* At the cur, summation of values can be calculated through the dfs search 

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
class MapSum:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        _root = lambda : defaultdict(_root)
        self.root = _root()

    def insert(self, key: str, val: int) -> None:
        cur = self.root
        for c in key:
            cur = cur[c]
        cur[True] = val

    def sum(self, prefix: str) -> int:
        cur = self.root
        for c in prefix:
            if c not in cur.keys():
                return 0
            cur = cur[c]

        def _dfs(cur, loc):
            res = loc
            for k in cur.keys():
                if isinstance(k, str):
                    res += _dfs(cur[k], loc)
            res += cur[True] if True in cur else 0
            return res

        ans = _dfs(cur=cur, loc=0)
        return ans
```

</div>
