---
title: "979.Distribute Coins in Binary Tree"
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
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
import sys
from binarytree import Node
sys.path.append("/home/swyoo/algorithm/")
from utils.verbose import buildtree
```

</div>

# 979. Distribute Coins in Binary Tree

## Idea
우리의 목표는 결국 모든 노드마다 코인이 하나씩 있도록 분배하는데 코인을 몇번 넘겨주냐를 구하는 것.
노드마다 남게되는 코인수를 `return` 하자. <br>
이 과정에서 `ans`를 업데이트 하면 된다. <br>
결국 모든 코인이 동일하게 1개씩 분배되기 때문에 (분배되지 않는 경우는 없음)
`left` 트리에서 `right` 트리에서 각각 필요한 수, 또는 남는 수만큼씩을 `ans`에 업데이트하면 된다. 
(이 부분이 생각하기 까다롭다. `left, mid, right` 에 대해 경우의 수를 나눠 생각해보면 <br>
`ans += (abs(left) + abs(right))` 만 해주면 됨을 알 수 있다.)


<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
class Solution:
    def distributeCoins(self, root: Node) -> int:
        ans = 0
        def dfs(cur: Node):
            """ return remainder coins, if deficient, negative value.
            update ans value through search process. """
            nonlocal ans
            if not cur:
                return 0
            left = dfs(cur.left)
            right = dfs(cur.right)
            mid = cur.value - 1
            rmd = left + right + mid
            ans += (abs(left) + abs(right))
            return rmd
        dfs(root)
        return ans
    
sol = Solution()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[10]:
</div>

<div class="input_area" markdown="1">

```python
null = None
A = [1,0,0,null,3]
# A = [1,0,2]
root = buildtree(A)
root.pprint()
print("ans:", sol.distributeCoins(root))
```

</div>

{:.output_stream}

```

  __1
 /   \
0     0
 \
  3

ans: 4

```
