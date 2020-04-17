---
title: "934.Shortest Bridge"
excerpt: "Find shortest bridge between two islands "

categories:
  - algorithms
tags:
  - BFS
  - DFS
use_math: true
last_modified_at: 2020-2-25
toc: true
toc_sticky: true
toc_label: "Contents"
toc_icon: "cog"
---

[leetcode](https://leetcode.com/problems/shortest-bridge/)
[discuss](https://leetcode.com/problems/shortest-bridge/discuss/189440/Python-concise-DFS-and-BFS-in-1-solution)

## Note
two island must exist in inputs.

## Key idea
Using DFS or BFS  in order to mark a `1st` island.  
When we mark first island, we can set seeds to start BFS in order to find `2nd` island.  
Just go through BFS search until finishing, and find minimum level among them.

**python implementation**

```python
from collections import defaultdict, deque 
import numpy as np
import sys

MAX = 1e+8

class Solution(object):
    def shortestBridge(self, A):
        """
        :type A: List[List[int]]
        :rtype: int
        """
        def isStart(i, j, M, N):
            """ returns True if (i,j) is edge of an island. """
            startable = [A[x][y] for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)] if 0 <= x < M and 0 <= y < N]
            return True if 0 in startable else False
        
        def adj(i, j):
            for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:                            
                if 0 <= x < M and 0 <= y < N:
                    yield x, y
        def discover():
            """ returns a discovered point in an island for the first time. """
            for i in range(M):
                for j in range(N):
                    if A[i][j] == 1:
                        return i, j
                    
        def BFS(i, j, seen):
            loc = MAX
            seen[(i,j)] = True 
            Q = deque([(i,j, 0)])
            while Q:
                r, c, lv = Q.popleft()
                # print(" - pop({},{},{})".format(r,c, lv), end="")
                
                if A[r][c] == 1 and lv > 0:
                    print("====>>> finish at lv={}".format(lv - 1))
                    loc = lv - 1 
                    return loc
                
                # print(">>seen as follows: {}".format(seen.keys()))
                for x, y in adj(r, c):
                    # print("  >search A[{}][{}]".format(x, y))
                    if (x, y) not in seen:
                        # print("  >explore A[{}][{}]".format(x, y))
                        seen[(x,y)] = True
                        Q.append((x, y, lv + 1))
                        # print("    >append ({},{},{})]".format(x,y, lv+1))
            return loc
        
        def markAnisland_v1(i, j, seen):
            """ BFS style: marks an island by seen """
            seen[(i, j)] = True
            dq = deque([(i, j)])
            while dq:
                r, c = dq.popleft()
                
                for x, y in adj(r, c):
                    if (x, y) not in seen and A[x][y] == 1:
                        seen[(x, y)] = True
                        dq.append((x, y))
        
        def markAnisland_v2(i, j, seen):
            """ DFS style: marks an island by seen 
            also, seen.keys() with isStart() 
            can be used as finding seeds for efficeint BFS later. """
            seen[(i, j)] = True # mark
            for x, y in adj(i, j):
                if (x, y) not in seen and A[x][y] == 1:
                    markAnisland_v2(x, y, seen)
            
        ans = MAX
        M, N = len(A), len(A[0])
        
        # preprocessing: use DFS or BFS
        # 1. fisrt island should be checked by seen.
        # 2. seen.keys() and isStart() can be used 
        # 		as seed positions of first island for BFS serach.
        seen = defaultdict(tuple)
        # markAnisland_v1(*discover(), seen) # BFS style
        markAnisland_v2(*discover(), seen) # DFS style
    
        
        # visualization of seen 
        np_seen = np.zeros(shape=(M, N), dtype=np.bool)
        for i, j in seen.keys():
            np_seen[i,j] = True
        print(np_seen)
        
        for i, j in seen.keys():
            if A[i][j] == 1 and isStart(i, j, M, N):
                print("-"*10, 'BFS', "-"*10)
                print("start at ({},{})".format(i,j))
                loc = BFS(i, j, seen.copy())          
                ans = min(ans, loc)
        return ans 

sol = Solution()
```

## Report 
  문제를 풀때,  한  island 가장가리에서 level을 증가시켜가며 BFS를 시작하여 다른 island에 도달하면 종료시키고, level을 return 하는 방식으로 sub문제들을 풀고, 가장 optimal한 답을 찾는 방식으로 접근하였다.
 내가 헷갈렸던 점은 가장 자리를 찾고 출발한 것은 좋았는데, 다른 island로 도달할때의 조건인 A의  element 1이 내가 출발한 island의 1 인지 다른 island의 1 인지 알 수 없다는 딜레마였다. 
 따라서, 한번 방문한 island는 또 방문하지 않도록  DFS나 BFS를 통한 둘 중 하나의 island에만 marking을 해놓고(seen dictionary를 사용), marking된 island의 외곽에서 출발하여 다른 island에 도달하면 종료하는 방식으로 알고리즘을 디자인하였다.

**생각하기 어려웠던 점**: BFS 또는 DFS를 2번 해야한다는 점!

























