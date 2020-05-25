---
title: "1315.Sum of Nodes with Even-Valued Grandparent"
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
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
from binarytree import Node, build
from collections import deque
```

</div>

# Sum of Nodes with Even-Valued Grandparent

[leetcode](https://leetcode.com/problems/sum-of-nodes-with-even-valued-grandparent/)

> <span style="color:red">Tip</span>: we can use `nonlocal ans` declared in `dfs` function instead of using `self.ans`. <br>


<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
class Solution:
    def __init__(self):
        self.ans = 0
    def sumEvenGrandparent(self, root: Node) -> int:
        # root.pprint()
        def dfs(cur: Node, pars:list):
            # nonlocal ans
            if not cur:
                return
            gpv = -1
            if len(pars) == 2:
                gpv = pars[0]
                pars.pop(0)
            dfs(cur.left, pars + [cur.value])
            dfs(cur.right, pars + [cur.value])
            if gpv % 2 == 0:
                self.ans += cur.value
                # ans += cur.value
        dfs(root, [])
        return self.ans

sol = Solution()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[8]:
</div>

<div class="input_area" markdown="1">

```python
null = None
A = [6, 7, 8, 2, 7, 1, 3, 9, null, 1, 4, null, null, null, 5]
root = build(A)
root.pprint()
```

</div>

{:.output_stream}

```

      ______6__
     /         \
    7__         8
   /   \       / \
  2     7     1   3
 /     / \         \
9     1   4         5


```

<div class="prompt input_prompt">
In&nbsp;[9]:
</div>

<div class="input_area" markdown="1">

```python
print(sol.sumEvenGrandparent(root))
```

</div>

{:.output_stream}

```
18

```

## Submitted Code

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def __init__(self):
        self.ans= 0
    def sumEvenGrandparent(self, root: TreeNode) -> int:
        # root.pprint()
        def dfs(cur: TreeNode, pars:list):
            if not cur:
                return
            gpv = -1
            if len(pars) == 2:
                gpv = pars[0]
                pars.pop(0)
            dfs(cur.left, pars + [cur.val])
            dfs(cur.right, pars + [cur.val])
            if gpv % 2 == 0:
                self.ans += cur.val
        dfs(root, [])
        return self.ans
```

</div>

## Discuss

I cited a code in the [document](https://leetcode.com/problems/sum-of-nodes-with-even-valued-grandparent/discuss/480981/Simple-Python-3-DFS-solution-beats-99.38).

<div class="prompt input_prompt">
In&nbsp;[16]:
</div>

<div class="input_area" markdown="1">

```python
class Solution:
    def sumEvenGrandparent(self, root: Node) -> int:
        def dfs(cur: Node, par: Node, gpar: Node):
            if not cur:
                return
            nonlocal ans
            # ans value in updated in the inorder.
            if par and gpar and gpar.value % 2 == 0:  
                ans += cur.value
            dfs(cur.left, cur, par)
            dfs(cur.right, cur, par)

        ans = 0
        dfs(root, None, None)
        return ans
    
sol = Solution()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[17]:
</div>

<div class="input_area" markdown="1">

```python
null = None
A = [6, 7, 8, 2, 7, 1, 3, 9, null, 1, 4, null, null, null, 5]
root = build(A)
root.pprint()
print(sol.sumEvenGrandparent(root))
```

</div>

{:.output_stream}

```

      ______6__
     /         \
    7__         8
   /   \       / \
  2     7     1   3
 /     / \         \
9     1   4         5

18

```
