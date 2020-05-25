---
title: "1080.Insufficient Nodes in Root to Leaf Paths"
excerpt: "Practice to use DFS"
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

# 1080. Insufficient Nodes in Root to Leaf Paths
[leetcode](https://leetcode.com/problems/insufficient-nodes-in-root-to-leaf-paths/)

## Idea
`root`에서 `leaf` 노드 까지 도달했을때 `limit - cumulative sum` 값을 구하고, <br> 
postorder 방식으로 search 하면서 internal node들에 들해서 update한다. <br>
(cur를 기준으로 cur.left와 cur.right둘다 없다면, 그 노드는 `None`으로 만들어 삭제)

<div class="prompt input_prompt">
In&nbsp;[39]:
</div>

<div class="input_area" markdown="1">

```python
from typing import List
from binarytree import build, Node, tree
import random
```

</div>

<div class="prompt input_prompt">
In&nbsp;[96]:
</div>

<div class="input_area" markdown="1">

```python
class Solution:
    def sufficientSubset(self, cur, limit):
        if cur.left == cur.right:
            return None if limit > cur.value else cur
        if cur.left:
            cur.left = self.sufficientSubset(cur.left, limit - cur.value)
        if cur.right:
            cur.right = self.sufficientSubset(cur.right, limit - cur.value)
        return cur if cur.left or cur.right else None
sol = Solution()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[97]:
</div>

<div class="input_area" markdown="1">

```python
null = None
A, limit = [1,2,-3,-5,null,4,null], -1
```

</div>

<div class="prompt input_prompt">
In&nbsp;[98]:
</div>

<div class="input_area" markdown="1">

```python
root = build(A)
root.pprint()
root = sol.sufficientSubset(root, limit)
root.pprint()
```

</div>

{:.output_stream}

```

     1__
    /   \
  _2     -3
 /      /
-5     4


1__
   \
    -3
   /
  4


```

<div class="prompt input_prompt">
In&nbsp;[107]:
</div>

<div class="input_area" markdown="1">

```python
root = tree(height=4)
root.pprint()
limit = 90
print("limit:", limit)
ans = sol.sufficientSubset(root, limit)
ans.pprint() if ans else None
```

</div>

{:.output_stream}

```

              ___________18______________
             /                           \
     _______16_____                _______26___
    /              \              /            \
  _2___          ___20          _30___         _9__
 /     \        /     \        /      \       /    \
27     _1      29      4      14      _12    21     19
      /  \       \           /       /             /
     25   8       5         3       13            6

limit: 90

18______________
                \
          _______26
         /
       _30___
      /      \
     14      _12
    /       /
   3       13


```

<div class="prompt input_prompt">
In&nbsp;[108]:
</div>

<div class="input_area" markdown="1">

```python
18 + 26 + 30 + 14 + 3
```

</div>




{:.output_data_text}

```
91
```


