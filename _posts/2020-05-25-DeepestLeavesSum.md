---
title: "1302.Deepest Leaves Sum"
excerpt: "practice to DFS"
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
import sys
sys.path.append("/home/swyoo/algorithm/")
from utils.verbose import buildtree
from binarytree import Node
```

</div>

# 1302. Deepest Leaves Sum

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
class Solution:
    def deepestLeavesSum(self, root: Node, verbose=False) -> int:
        maxlv, cands = 0, []
        def dfs(cur, lv):
            nonlocal maxlv
            if cur.left == cur.right == None:
                if lv >= maxlv:
                    maxlv = lv
                    cands.append((cur.value, lv))
                return
            if cur.left:
                dfs(cur.left, lv + 1)
            if cur.right:
                dfs(cur.right, lv + 1)
        dfs(root, 0)
        if verbose:
            root.pprint()
            print(cands, maxlv)
        return sum(x for x, lv in cands if lv == maxlv)

sol = Solution()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
null = None
A = [1,2,3,4,5,null,6,7,null,null,null,null,8]
print(sol.deepestLeavesSum(buildtree(A), verbose=True))
```

</div>

{:.output_stream}

```

      __1
     /   \
    2     3
   / \     \
  4   5     6
 /           \
7             8

[(7, 3), (8, 3)] 3
15

```
