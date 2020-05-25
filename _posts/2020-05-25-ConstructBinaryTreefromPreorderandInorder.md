---
title: "105.Construct Binary Tree from Preorder and Inorder Traversal"
excerpt: "practice of binary tree traversal"
categories:
 - algorithms
tags:
 - datastructure
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
In&nbsp;[13]:
</div>

<div class="input_area" markdown="1">

```python
from binarytree import tree, build, Node
from typing import List
```

</div>

# 105. Construct Binary Tree from Preorder and Inorder Traversal

<div class="prompt input_prompt">
In&nbsp;[18]:
</div>

<div class="input_area" markdown="1">

```python
# toy example
root = tree(height=3)
root.pprint()
print(root.preorder)
print(root.inorder)
```

</div>

{:.output_stream}

```

         _____10_____
        /            \
    ___9           ___7
   /    \         /    \
  8      14      12     6
 / \       \       \
1   13      0       4

[Node(10), Node(9), Node(8), Node(1), Node(13), Node(14), Node(0), Node(7), Node(12), Node(4), Node(6)]
[Node(1), Node(8), Node(13), Node(9), Node(14), Node(0), Node(10), Node(12), Node(4), Node(7), Node(6)]

```

## Idea 
I cited an idea from [discuss](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/discuss/34579/Python-short-recursive-solution.) 

I cited [a user](https://leetcode.com/sys)'s comment from the leetcode.
>Looking at preorder traversal, the first value (node 1) must be the root. <br>
Then, we find the index of root within in-order traversal, and split into two sub problems

Therefore, the `tree` can be constructed in the `preorder` by referring to the given `inorder`. 

![](https://leetcode.com/uploads/files/1486248260436-screenshot-2017-02-04-17.44.08.png)

>### Tip
a built-in function of `list` is `list.index`, which find the index from the given value.

<div class="prompt input_prompt">
In&nbsp;[33]:
</div>

<div class="input_area" markdown="1">

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Node:
        if not inorder:  # if inorder is empty, return
            return None
        idx = inorder.index(preorder.pop(0))
        root = Node(inorder[idx])
        root.left = self.buildTree(preorder, inorder[:idx])
        root.right = self.buildTree(preorder, inorder[idx+1:])
        return root
sol = Solution()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[34]:
</div>

<div class="input_area" markdown="1">

```python
root.pprint()
preorder = list(map(lambda x: x.value, root.preorder))
inorder = list(map(lambda x: x.value, root.inorder))
print(preorder)
print(inorder)
print("-"*50)
ans = sol.buildTree(preorder, inorder)
ans.pprint()
```

</div>

{:.output_stream}

```

         _____10_____
        /            \
    ___9           ___7
   /    \         /    \
  8      14      12     6
 / \       \       \
1   13      0       4

[10, 9, 8, 1, 13, 14, 0, 7, 12, 4, 6]
[1, 8, 13, 9, 14, 0, 10, 12, 4, 7, 6]
--------------------------------------------------

         _____10_____
        /            \
    ___9           ___7
   /    \         /    \
  8      14      12     6
 / \       \       \
1   13      0       4


```
