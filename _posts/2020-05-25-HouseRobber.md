---
title: "337.House Robber III"
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
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
import sys
sys.path.append("/home/swyoo/algorithm/")
from utils.verbose import logging_time, buildtree
from binarytree import build, tree, Node
```

</div>

# House Robber III

It will automatically contact the police <span style="color:red"> if two directly-linked houses were broken into on the same night. </span>

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
null = None
A = [3, 2, 3, null, 3, null, 1]
buildtree(A).pprint()
```

</div>

{:.output_stream}

```

  __3
 /   \
2     3
 \     \
  3     1


```

Output: 7  <br>
Explanation: Maximum amount of money the thief can rob = 3 + 3 + 1 = 7. <br>

## DFS search

Note that 
* If current node is robbed, both left and right children must not be robbed(skipped). 
* If current nodes are not robbed, just merge optimal left and right values(overlapping subproblems).

Let optimal value of a cur(as a root node) of a subtree be $G^*$. <br>

Resursive formula as follows such that <br>
* $*$ means the optimal case.
* $i$ of $G^i$ determines rob the node $i$ or not. 
* $G_{left}$ or $G_{right}$ determines the left node or right node.
$$
\begin{align}
G_i^* &= max(G_i^1, G_i^2) \\
G_i^1 &= G_{left}^* + G_{right}^* , \text{where } G_{case} = max(G_{left}^1, G_{right}^2)* \\
G_i^2 &= G_{left}^* + G_{right}^* + i_{val} \\
\end{align}
$$

It takes $O(n!)$ because entry $i$ takes $O(i!)$ through recursive call.

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
class Solution1:
    @logging_time
    def rob(self, root: Node) -> int:
        if not root: return 0
        def dfs(cur, rob:bool) -> int:
            if not cur:
                return 0
            if rob:
                return cur.value + dfs(cur.left, rob=False) + dfs(cur.right, rob=False)
            leftmax = max(dfs(cur.left, rob=True), dfs(cur.left, rob=False))
            rightmax = max(dfs(cur.right, rob=True), dfs(cur.right, rob=False))
            return leftmax + rightmax

        return max(dfs(root, rob=True), dfs(root, rob=False))
sol1 = Solution1()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
print(sol1.rob(buildtree(A), verbose=True))
```

</div>

{:.output_stream}

```
WorkingTime[rob]: 0.11730 ms
7

```

## DFS search + DP

This problem satisfies optimal substructure property, overlapping subproblems. <br>
Therefore, dynamic programming is possible. <br>
Time complexity can be improved as follows: $O(n)$ <br>
This is because the number of all entries is $n$, each entry takes $O(1)$. <br>
(recall that $ G_i^* = max(G_i^1, G_i^2)$, just select a $max$ value between first or second cases.)

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
class Solution2:
    @logging_time
    def rob(self, root:Node) -> int:
        if not root: return 0
        seen = {}
        def dfs(cur, rob:bool) -> int:
            if (cur, rob) in seen:
                return seen[cur, rob]
            if not cur:
                return 0
            if rob:
                seen[(cur, rob)] = cur.value + dfs(cur.left, rob=False) + dfs(cur.right, rob=False)
                return seen[(cur, rob)]
            leftmax = max(dfs(cur.left, rob=True), dfs(cur.left, rob=False))
            rightmax = max(dfs(cur.right, rob=True), dfs(cur.right, rob=False))
            seen[(cur, rob)] = leftmax + rightmax
            return seen[(cur, rob)]

        return max(dfs(root, rob=True), dfs(root, rob=False))
sol2 = Solution2()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
print(sol1.rob(buildtree(A) ,verbose=True))
print(sol2.rob(buildtree(A) ,verbose=True))
```

</div>

{:.output_stream}

```
WorkingTime[rob]: 0.11992 ms
7
WorkingTime[rob]: 0.08249 ms
7

```

### Test using brinarytree module

<div class="prompt input_prompt">
In&nbsp;[7]:
</div>

<div class="input_area" markdown="1">

```python
root = tree(height=9, is_perfect=False)
# root.pprint()
print(len(root.values))
print(sol1.rob(root ,verbose=True))
print(sol2.rob(root, verbose=True))
```

</div>

{:.output_stream}

```
1012
WorkingTime[rob]: 216.88080 ms
157917
WorkingTime[rob]: 11.42621 ms
157917

```

# Reference
[1] [discuss](https://leetcode.com/problems/house-robber-iii/discuss/79330/Step-by-step-tackling-of-the-problem)
