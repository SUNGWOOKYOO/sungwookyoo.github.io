---
title: "79.Word Search"
excerpt: "Given a 2D board and a word, find if the word exists in the grid."

categories:
  - algorithms
tags:
  - DFS
use_math: true
last_modified_at: 2020-02-29
toc: true
toc_sticky: true
toc_label: "Contents"
toc_icon: "cog"
header:
  overlay_image: /assets/images/tips.png
  overlay_filter: 0.5
  caption: python tips
  actions:
    - label: "leetcode"
      url: "https://leetcode.com/problems/word-search/"
    - discuss: "discuss"
      url: "https://leetcode.com/problems/word-search/discuss/27660/Python-dfs-solution-with-comments."
---

Given a 2D board and a word, find if the word exists in the grid.

The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring.   
The same letter cell may not be used more than once.

```shell
Example:

board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

Given word = "ABCCED", return true.
Given word = "SEE", return true.
Given word = "ABCB", return false.
```



## Key idea

Using DFS, search the `board` until emerging a given `word`.

Note that **the same letter cell may not be used more than once.**

If we enumerate all cases, it takes too much time. 

So, we have to prune some cases as follows. 

* Only search for substring of the given word starting from beginning.
  * use `pos` to count substring of `word`.
* Use `seen` to avoid revisit.

2 cases can be used in code as follows.

```python
if seen[(i, j)] or (board[i][j] != word[pos]): # pruning cases.
	return False
```

<div style="background-color:gray"> <summary> <font color=red> Warning </font></summary> <p> We should use carefully when we use `seen`. This is because the algorithm sometimes need rollback when it goes to wrong direction. The detail will be described in next section - example.</p></div>
### Outline of my code

```python
def dfs(i, j, ...):
    if pos == len(word):
        # terminating case; find all characters of the given word.
        return True
    if seen[(i, j)] or (board[i][j] != word[pos]): 
        # pruning cases.
        return False
    
    # visiting time
    seen[(i, j)] = True # mark 
    loc = False
    for x, y in adj(i, j):
        loc = loc or dfs(x, y) # recursive call
    
    # finishing time
    # IMPORTANT: roll back!
    seen[(i, j)] = False    
    return loc
```



import libraries

```python
import numpy as np
from termcolor import colored
from collections import defaultdict
import time

def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        elapsed_time = (time.time() - start_time) * 1e3
        print("WorkingTime[{}]: {:.5f} ms".format(original_fn.__name__, elapsed_time))
        return result
    return wrapper_fn
```



## DFS - My Code 

```python
from collections import defaultdict
class Solution(object):
    def __init__(self):
        # bool value is immutable.
        # However, if it is object variable, it can be mutable.
        self.ans = False 
    @logging_time
    def exist(self, board, word, verbose=False):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        m, n = len(board), len(board[0])
        def adj(i, j):
            for x, y in [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]:
                if 0 <= x < m and 0 <= y < n:
                    yield x, y
         
        def dfs(i, j, pos, seen):
            if pos == len(word):
                if verbose: print(colored("*********** >> return True ***********", 'red'))
                return True
            if seen[(i, j)] or (board[i][j] != word[pos]): # pruning cases.
                if verbose: print("~borad[{}][{}] pruned".format(i, j))
                return False
                
            seen[(i, j)] = True
            if verbose: 
                visualization = np.array(board)
                for r, c in seen.keys():
                    if seen[(r,c)] == True:
                        visualization[r, c] = "#"
                print("{}, visited path of characters: {}, word:{}"
                      .format(visualization, word[:pos] + board[i][j], word[:pos+1]))
                
            loc = False
            for x, y in adj(i, j):
                loc = loc or dfs(x, y, pos + 1, seen)
            seen[(i, j)] = False # important: if we went to wrong dirction, we can rollback.
            if verbose: print("{} at board[{}][{}]".format(colored("finish", "red"), i, j))
            return loc
            
        for r in range(m):
            for c in range(n):
                # if any path that all characters are matched with words exist, return True.
                # (board[r][c] == word) is exception because adj[0][0] is [] when size=[1,1]
                if verbose: print("=-=-= start from board[{}][{}] =-=-=".format(r, c))
                if dfs(r, c, 0, defaultdict(lambda: False)) or (board[r][c] == word):
                    return True
        return False

sol1 = Solution()
```

```python
# running example
board = [["A","B","C","E"],["S","F","E","S"],["A","D","E","E"]]
word = "ABCESEEEFS"
print("sol1:", sol1.exist(board, word, True))
"""
=-=-= start from board[0][0] =-=-=
[['#' 'B' 'C' 'E']
 ['S' 'F' 'E' 'S']
 ['A' 'D' 'E' 'E']], visited path of characters: A, word:A
~borad[1][0] pruned
[['#' '#' 'C' 'E']
 ['S' 'F' 'E' 'S']
 ['A' 'D' 'E' 'E']], visited path of characters: AB, word:AB
~borad[1][1] pruned
~borad[0][0] pruned
[['#' '#' '#' 'E']
 ['S' 'F' 'E' 'S']
 ['A' 'D' 'E' 'E']], visited path of characters: ABC, word:ABC
[['#' '#' '#' 'E']
 ['S' 'F' '#' 'S']
 ['A' 'D' 'E' 'E']], visited path of characters: ABCE, word:ABCE
~borad[0][2] pruned
~borad[2][2] pruned
~borad[1][1] pruned
[['#' '#' '#' 'E']
 ['S' 'F' '#' '#']
 ['A' 'D' 'E' 'E']], visited path of characters: ABCES, word:ABCES
[['#' '#' '#' '#']
 ['S' 'F' '#' '#']
 ['A' 'D' 'E' 'E']], visited path of characters: ABCESE, word:ABCESE
~borad[1][3] pruned
~borad[0][2] pruned
finish at board[0][3]
[['#' '#' '#' 'E']
 ['S' 'F' '#' '#']
 ['A' 'D' 'E' '#']], visited path of characters: ABCESE, word:ABCESE
~borad[1][3] pruned
[['#' '#' '#' 'E']
 ['S' 'F' '#' '#']
 ['A' 'D' '#' '#']], visited path of characters: ABCESEE, word:ABCESEE
~borad[1][2] pruned
~borad[2][1] pruned
~borad[2][3] pruned
finish at board[2][2]
finish at board[2][3]
~borad[1][2] pruned
finish at board[1][3]
finish at board[1][2]
~borad[0][1] pruned
[['#' '#' '#' '#']
 ['S' 'F' 'E' 'S']
 ['A' 'D' 'E' 'E']], visited path of characters: ABCE, word:ABCE
[['#' '#' '#' '#']
 ['S' 'F' 'E' '#']
 ['A' 'D' 'E' 'E']], visited path of characters: ABCES, word:ABCES
~borad[0][3] pruned
[['#' '#' '#' '#']
 ['S' 'F' 'E' '#']
 ['A' 'D' 'E' '#']], visited path of characters: ABCESE, word:ABCESE
~borad[1][3] pruned
[['#' '#' '#' '#']
 ['S' 'F' 'E' '#']
 ['A' 'D' '#' '#']], visited path of characters: ABCESEE, word:ABCESEE
[['#' '#' '#' '#']
 ['S' 'F' '#' '#']
 ['A' 'D' '#' '#']], visited path of characters: ABCESEEE, word:ABCESEEE
~borad[0][2] pruned
~borad[2][2] pruned
[['#' '#' '#' '#']
 ['S' '#' '#' '#']
 ['A' 'D' '#' '#']], visited path of characters: ABCESEEEF, word:ABCESEEEF
~borad[0][1] pruned
~borad[2][1] pruned
[['#' '#' '#' '#']
 ['#' '#' '#' '#']
 ['A' 'D' '#' '#']], visited path of characters: ABCESEEEFS, word:ABCESEEEFS
*********** >> return True ***********
finish at board[1][0]
finish at board[1][1]
finish at board[1][2]
finish at board[2][2]
finish at board[2][3]
finish at board[1][3]
finish at board[0][3]
finish at board[0][2]
finish at board[0][1]
finish at board[0][0]
WorkingTime[exist]: 7.84993 ms
sol1: True
"""
```

## Discuss 

there is good solution in [leecode](https://leetcode.com/problems/word-search/discuss/27660/Python-dfs-solution-with-comments.).

This is same approah, but use less memories.

```python
class Solution(object):
    @logging_time
    def exist(self, board, word, verbose=False):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        ans = False
        m, n = len(board), len(board[0])
        def adj(i, j):
            for x, y in [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]:
                if 0 <= x < m and 0 <= y < n:
                    yield x, y
         
        def dfs(i, j, wd):
            if len(wd) == 0:
                if verbose: print(colored("*********** >> return True ***********", 'red'))
                # all characters are matched with given word.
                return True
            if wd[0] != board[i][j]: 
                # pruning non-matched cases while DFS searching. 
                if verbose: print("search passes going through board[{}][{}] are pruned.".format(i, j))
                return False
            # conceal current character until finshing time.
            tmp = board[i][j]
            board[i][j] = '#'
            if verbose: print("{}, remainder characters: {}".format(np.array(board), wd[1:]) ,end='\n'+'='*30+'\n')
            loc = False
            for x, y in adj(i, j):
                loc = loc or dfs(x, y, wd[1:])
            
            # at finishing time, restore the conceal word.
            board[i][j] = tmp
            if verbose: print("{} board[{}][{}], return {}\n".format(colored("finish", 'red'),i, j, loc), np.array(board), end='\n'+'='*30+'\n')
            return loc
        
        for r in range(m):
            for c in range(n):
                # if any path that all characters are matched with words exist, return True.
                # (board[r][c] == word) is exception because adj[0][0] is [] when size=[1,1]
                if dfs(r, c, word) or (board[r][c] == word):
                    return True
        return False

sol2 = Solution()
```

