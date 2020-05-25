---
title: "How to implement and visualize Binary Tree "
excerpt: "tips for making use of binary tree"
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
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
import random
from binarytree import build, tree, bst, heap, Node
plot = lambda x: build(a).pprint()
```

</div>

# Build Binary Tree 

## Binary tree library

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
size = 4
root = tree(height=size, is_perfect=False)
print("A={}, |A|={}".format(root.values, len(root.values)))
root.pprint()
```

</div>

{:.output_stream}

```
A=[28, 30, 25, 24, 0, 13, 14, 20, 23, 26, 18, 10, 21, 4, 8, 7, None, 2, None, None, 11, 9, 3, None, None, None, 29, None, 16, 12], |A|=30

               ________________28____________
              /                              \
       ______30______                  _______25_____
      /              \                /              \
    _24__         ____0__           _13            ___14___
   /     \       /       \         /   \          /        \
  20      23    26        18      10    21       4         _8
 /       /        \      /  \             \       \       /
7       2          11   9    3             29      16    12


```

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
A = [1,2,3,4,-99,-99,7,8,9,-99,-99,12,13,-99,14]
build(A).pprint()  # use binary tree libray
```

</div>

{:.output_stream}

```

        ____________1__________
       /                       \
    __2_____                ____3____
   /        \              /         \
  4        _-99_        _-99         _7
 / \      /     \      /    \       /  \
8   9   -99     -99   12     13   -99   14


```

## Build binary tree for Leetcode problems.

If I solve a problem related to binary tree in [leetcode](https://leetcode.com/), <br>
<span style="color:red">the representatation of binary tree is somewhat different. </span><br>
I will show you as follows. <br>
(If a internal node is `null`, it makes error when indexing).

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
null = None
A = [1,2,3,4,null,null,7,8,9,null,14]
# build(A).pprint() # Error!
```

</div>

Therefore I implemented it by referring to [StefanPochmann
's post](https://leetcode.com/problems/recover-binary-search-tree/discuss/32539/Tree-Deserializer-and-Visualizer-for-Python)

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
def buildtree(A):
    if not A: return None
    nodes = [None if x == None else Node(x) for x in A]
    kids = nodes[::-1]
    root = kids.pop()
    for node in nodes:
        if node:
            if kids: node.left  = kids.pop()
            if kids: node.right = kids.pop()
    return root
```

</div>

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
print(A)
buildtree(A).pprint()
```

</div>

{:.output_stream}

```
[1, 2, 3, 4, None, None, 7, 8, 9, None, 14]

        1
       / \
    __2   3
   /       \
  4         7
 / \         \
8   9         14


```

<div class="prompt input_prompt">
In&nbsp;[7]:
</div>

<div class="input_area" markdown="1">

```python
buildtree([5,4,8,11,null,17,4,7,null,null,null,5]).pprint()
```

</div>

{:.output_stream}

```

       5___
      /    \
    _4     _8__
   /      /    \
  11     17     4
 /             /
7             5


```

# Reference
[1] [binary tree library in python](https://pypi.org/project/binarytree/)<br>
[2] [StefanPochmann
's post](https://leetcode.com/problems/recover-binary-search-tree/discuss/32539/Tree-Deserializer-and-Visualizer-for-Python) <br>
