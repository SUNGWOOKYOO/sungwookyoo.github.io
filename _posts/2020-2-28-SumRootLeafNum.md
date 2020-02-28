---
title: "129.Sum Root to Leaf Numbers"
excerpt: "Find the total sum of all root-to-leaf numbers."

categories:
  - algorithms
tags:
  - DFS
  - BFS
use_math: true
last_modified_at: 2020-02-28
toc: true
toc_sticky: true
toc_label: "Contents"
toc_icon: "cog"
header:
  overlay_image: /assets/images/algorithms/algo.png
  overlay_filter: 0.5
  caption: think util some idea is emerged.
  actions:
    - label: "leetcode"
      url: "https://leetcode.com/problems/sum-root-to-leaf-numbers/"
---

# DFS

My code is simple.  
We can easily solve this problem using DFS.  
## Key idea
When we visit a node using **DFS**,  <br> 
we have to collect visited `node.val` by accumulating previously visited nodes.

**How to implement this? <br>**
* `num` as a argument of DFS function, and update it recursively. 
    * `num` should be immutable to prevent (inner)recursive function from altering num.
        ```python
        def dfs(u, num=''):
               ...
        ```
* At the finish time of a node, collect the recursively updated num. 
    * please note that we should ignore the finishing time of non-leaf nodes. 
        ```python
         if u.left == None and u.right == None:
                nums.append(num)
        
        ```
        

### python implementation

````python
from collections import defaultdict, deque

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def sumNumbers(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root is None: return 0
        # Actually, we don't need `seen`.
        # This is because we do not revisit a node once the node was visited before.
        seen = defaultdict(TreeNode)
        nums = []
        def dfs(u, num=''):
            """ update num using DFS search in order to collect numbers to sum. 
            Note:
                num should be immutable, so I used 'str' datastructure in python. """
            seen[u] = True
            num += str(u.val) # update num by visiting. 
            
            for v in [u.left, u.right]:
                if v not in seen and v != None:
                    dfs(v, num)
            # at the finishing time, collect updated num if `u` is a leaf node.
            if u.left == None and u.right == None:
                nums.append(num)
            
        dfs(root)
        print(nums) # show collated numbers.
        ans = 0 
        for e in nums:
            ans += int(e)
        return ans           

sol = Solution()
````

```python
root = TreeNode(4)
root.left, root.right = TreeNode(9), TreeNode(0)
root.left.left, root.left.right = TreeNode(5), TreeNode(1)
sol.sumNumbers(root)
""" 
['495', '491', '40']
1026
"""
```

## The final code.

Actually, we don't need `seen`.  <br> 
This is because we do not revisit a node once the node was visited before.

```python
from collections import defaultdict, deque

class Solution(object):
    def sumNumbers(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root is None: return 0
        nums = []
        def dfs(u, num=''):
            num += str(u.val)
            for v in [u.left, u.right]:
                if v != None:
                    dfs(v, num)
            
            if u.left == None and u.right == None:
                nums.append(num)
            
        dfs(root)
        
        ans = 0 
        for e in nums:
            ans += int(e)
        return ans    
```

# Others
[discuss](https://leetcode.com/problems/sum-root-to-leaf-numbers/discuss/41383/Python-solutions-(dfs%2Bstack-bfs%2Bqueue-dfs-recursively).)

iterative version with DFS is possible. <br>
Also, BFS approach is possible.
