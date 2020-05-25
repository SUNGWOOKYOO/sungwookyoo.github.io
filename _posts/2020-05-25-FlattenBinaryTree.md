---
title: "114.Flatten Binary Tree to Linked List"
excerpt: "practice of DFS"
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
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
from binarytree import Node, build
```

</div>

# 114. Flatten Binary Tree to Linked List

**2 step solution**
1. `[cur, cur.left, cur.right]` order traversal to store the right linked list. 
2. flatten given tree from root by using the stored list. 

Let $|V|$ be $n$, the time complexity is $O(n)$

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
class Solution:
    def flatten(self, root: Node) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        order = []
        def func(cur:Node):
            if not cur:
                return
            order.append(cur)
            func(cur.left)
            func(cur.right)
        func(root)
        now = order[0]
        for x in order[1:]:
            now.right = x
            now.left = None
            now = now.right
sol = Solution()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
null = None
A = [1,2,5,3,4,null,6]
root = build(A)
root.pprint()
sol.flatten(root)
root.pprint()
```

</div>

{:.output_stream}

```

    __1
   /   \
  2     5
 / \     \
3   4     6


1
 \
  2
   \
    3
     \
      4
       \
        5
         \
          6


```
