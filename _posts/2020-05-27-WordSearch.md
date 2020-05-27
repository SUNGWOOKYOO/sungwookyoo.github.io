---
title: "79.Word Search"
excerpt: "Given a 2D board and a word, find if the word exist in the grid: backtracking, pruning practice."
categories:
 - algorithms
tags:
 - DFS
 - enumerate
use_math: true
last_modified_at: "2020-05-27"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/teaser.jpg
 overlay_filter: 0.5
---

# 79. Word Search

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

<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
import numpy as np
from termcolor import colored
from collections import defaultdict
import time
```

</div>

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        elapsed_time = (time.time() - start_time) * 1e3
        print("WorkingTime[{}]: {:.5f} ms".format(original_fn.__name__, elapsed_time))
        return result
    return wrapper_fn
```

</div>

# Mycode

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

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

</div>

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
board = \
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]
word = "ABCCED"
print("sol1:", sol1.exist(board, word, verbose=True))
```

</div>

{:.output_stream}

```
=-=-= start from board[0][0] =-=-=
[['#' 'B' 'C' 'E']
 ['S' 'F' 'C' 'S']
 ['A' 'D' 'E' 'E']], visited path of characters: A, word:A
~borad[1][0] pruned
[['#' '#' 'C' 'E']
 ['S' 'F' 'C' 'S']
 ['A' 'D' 'E' 'E']], visited path of characters: AB, word:AB
~borad[1][1] pruned
~borad[0][0] pruned
[['#' '#' '#' 'E']
 ['S' 'F' 'C' 'S']
 ['A' 'D' 'E' 'E']], visited path of characters: ABC, word:ABC
[['#' '#' '#' 'E']
 ['S' 'F' '#' 'S']
 ['A' 'D' 'E' 'E']], visited path of characters: ABCC, word:ABCC
~borad[0][2] pruned
[['#' '#' '#' 'E']
 ['S' 'F' '#' 'S']
 ['A' 'D' '#' 'E']], visited path of characters: ABCCE, word:ABCCE
~borad[1][2] pruned
[['#' '#' '#' 'E']
 ['S' 'F' '#' 'S']
 ['A' '#' '#' 'E']], visited path of characters: ABCCED, word:ABCCED
[31m*********** >> return True ***********[0m
[31mfinish[0m at board[2][1]
[31mfinish[0m at board[2][2]
[31mfinish[0m at board[1][2]
[31mfinish[0m at board[0][2]
[31mfinish[0m at board[0][1]
[31mfinish[0m at board[0][0]
WorkingTime[exist]: 0.83470 ms
sol1: True

```

# Discuss

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

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

</div>

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
board = \
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]
word = "ABCCED"
print("sol1:", sol1.exist(board, word, verbose=False))
sol2.exist(board, word, verbose=True)
```

</div>

{:.output_stream}

```
WorkingTime[exist]: 0.02527 ms
sol1: True
[['#' 'B' 'C' 'E']
 ['S' 'F' 'C' 'S']
 ['A' 'D' 'E' 'E']], remainder characters: BCCED
==============================
search passes going through board[1][0] are pruned.
[['#' '#' 'C' 'E']
 ['S' 'F' 'C' 'S']
 ['A' 'D' 'E' 'E']], remainder characters: CCED
==============================
search passes going through board[1][1] are pruned.
search passes going through board[0][0] are pruned.
[['#' '#' '#' 'E']
 ['S' 'F' 'C' 'S']
 ['A' 'D' 'E' 'E']], remainder characters: CED
==============================
[['#' '#' '#' 'E']
 ['S' 'F' '#' 'S']
 ['A' 'D' 'E' 'E']], remainder characters: ED
==============================
search passes going through board[0][2] are pruned.
[['#' '#' '#' 'E']
 ['S' 'F' '#' 'S']
 ['A' 'D' '#' 'E']], remainder characters: D
==============================
search passes going through board[1][2] are pruned.
[['#' '#' '#' 'E']
 ['S' 'F' '#' 'S']
 ['A' '#' '#' 'E']], remainder characters: 
==============================
[31m*********** >> return True ***********[0m
[31mfinish[0m board[2][1], return True
 [['#' '#' '#' 'E']
 ['S' 'F' '#' 'S']
 ['A' 'D' '#' 'E']]
==============================
[31mfinish[0m board[2][2], return True
 [['#' '#' '#' 'E']
 ['S' 'F' '#' 'S']
 ['A' 'D' 'E' 'E']]
==============================
[31mfinish[0m board[1][2], return True
 [['#' '#' '#' 'E']
 ['S' 'F' 'C' 'S']
 ['A' 'D' 'E' 'E']]
==============================
[31mfinish[0m board[0][2], return True
 [['#' '#' 'C' 'E']
 ['S' 'F' 'C' 'S']
 ['A' 'D' 'E' 'E']]
==============================
[31mfinish[0m board[0][1], return True
 [['#' 'B' 'C' 'E']
 ['S' 'F' 'C' 'S']
 ['A' 'D' 'E' 'E']]
==============================
[31mfinish[0m board[0][0], return True
 [['A' 'B' 'C' 'E']
 ['S' 'F' 'C' 'S']
 ['A' 'D' 'E' 'E']]
==============================
WorkingTime[exist]: 1.22833 ms

```




{:.output_data_text}

```
True
```



<div class="prompt input_prompt">
In&nbsp;[7]:
</div>

<div class="input_area" markdown="1">

```python
# exception
board = [["a"]]
word = "a"
print("sol1:", sol1.exist(board, word, verbose=False))
sol2.exist(board, word, verbose=True)
```

</div>

{:.output_stream}

```
WorkingTime[exist]: 0.00882 ms
sol1: True
[['#']], remainder characters: 
==============================
[31mfinish[0m board[0][0], return False
 [['a']]
==============================
WorkingTime[exist]: 0.15259 ms

```




{:.output_data_text}

```
True
```



# Report
seen list에 넣었다가 finish time에 삭제하면 될거 같다.

<div class="prompt input_prompt">
In&nbsp;[8]:
</div>

<div class="input_area" markdown="1">

```python
board = [["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","b"]]
word = "baaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
print("sol1:", sol1.exist(board, word))
print("sol2:", sol2.exist(board, word))
```

</div>

{:.output_stream}

```
WorkingTime[exist]: 2.99692 ms
sol1: True
WorkingTime[exist]: 2.47836 ms
sol2: True

```

<div class="prompt input_prompt">
In&nbsp;[9]:
</div>

<div class="input_area" markdown="1">

```python
board = [["A","B","C","E"],["S","F","E","S"],["A","D","E","E"]]
word = "ABCESEEEFS"
print("sol1:", sol1.exist(board, word, True))
sol2.exist(board, word, verbose=False)
```

</div>

{:.output_stream}

```
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
[31mfinish[0m at board[0][3]
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
[31mfinish[0m at board[2][2]
[31mfinish[0m at board[2][3]
~borad[1][2] pruned
[31mfinish[0m at board[1][3]
[31mfinish[0m at board[1][2]
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
[31m*********** >> return True ***********[0m
[31mfinish[0m at board[1][0]
[31mfinish[0m at board[1][1]
[31mfinish[0m at board[1][2]
[31mfinish[0m at board[2][2]
[31mfinish[0m at board[2][3]
[31mfinish[0m at board[1][3]
[31mfinish[0m at board[0][3]
[31mfinish[0m at board[0][2]
[31mfinish[0m at board[0][1]
[31mfinish[0m at board[0][0]
WorkingTime[exist]: 4.05359 ms
sol1: True
WorkingTime[exist]: 0.03242 ms

```




{:.output_data_text}

```
True
```



## Review

복습을 위해 다시 풀어 보았다. 

<div class="prompt input_prompt">
In&nbsp;[10]:
</div>

<div class="input_area" markdown="1">

```python
import sys, random
sys.path.append("/home/swyoo/algorithm/")
from typing import List
from utils.verbose import logging_time
from utils.generator import random2Dcharacters, randomString
```

</div>

### DFS with Pruning Version 1
주어진 word 길이가 $N$이라고하고, board shape를 $m, n$이라 하자. <br>
DFS를 사용하여 $O(4^N)$ 안에 구할 수 있다(모든 cases를 보게 되므로). <br>
주목할 점은 pruning을 사용하기 때문에 가능한 범위 내에서 모든 case를 보게된다. <br> 
pruning을 쓰지 않으면 당연히 `LTE`가 뜬다. <br>
전역 변수를 만들고, 한번이라도 단어의 끝에 도달하면 `True`가 된다.

<div class="prompt input_prompt">
In&nbsp;[11]:
</div>

<div class="input_area" markdown="1">

```python
class Solution1:
    @logging_time
    def exist(self, board: List[List[str]], word: str) -> bool:
        if not word: return False
        m, n = len(board), len(board[0])

        def adj(i, j):
            for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                if 0 <= x < m and 0 <= y < n:
                    yield x, y

        seen = set()
        ans = False
        def dfs(i, j, depth):
            nonlocal ans
            if depth == len(word):
                ans = True
                return 

            for x, y in adj(i, j):
                if board[x][y] == word[depth] and (x, y) not in seen:
                    seen.add((x, y))  # mark seen until finishing
                    dfs(x, y, depth + 1)
                    seen.discard((x, y))  # revert seen

        for i, rows in enumerate(board):
            for j, c in enumerate(rows):
                if c == word[0]:
                    seen.add((i, j))
                    dfs(i, j, 1)
                    seen.discard((i, j))
        return ans
    
sol1 = Solution1()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[12]:
</div>

<div class="input_area" markdown="1">

```python
board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
word = "ABCB"
import numpy as np
print(np.array(board))
sol1 = Solution1()
print(sol1.exist(board, word, verbose=True))
# if show: print(board[i][j], (i, j), depth)
```

</div>

{:.output_stream}

```
[['A' 'B' 'C' 'E']
 ['S' 'F' 'C' 'S']
 ['A' 'D' 'E' 'E']]
WorkingTime[exist]: 0.01621 ms
False

```

### DFS with Pruning Version 2

그런데, 더 개선가능 한 점이 있다. 다음과 같은 사실을 기억해보자!.
<span style="color:red">한번이라도 단어의 끝에 도달하면 True가 된다</span>. <br>
만약 worst case의 경우, 전역 변수로 DFS를 하게 된다면, <br>
word 끝까지 DFS에 성공해서 ans가 True가 되어도 다른 경우의 수까지 다시 recursive call하게 된다. <br>
따라서, 어떤 격자지점에서 하나의 경우라도 DFS 가 True가 된다면, recursion(재귀)를 끝내도록 디자인한다. <br>
이렇게 해야만 [leetcode](https://leetcode.com/problems/word-search/submissions/)에서 통과 되었다. <br>

<div class="prompt input_prompt">
In&nbsp;[13]:
</div>

<div class="input_area" markdown="1">

```python
class Solution2:
    @logging_time
    def exist(self, board: List[List[str]], word: str) -> bool:
        if not word: return False
        m, n = len(board), len(board[0])

        def adj(i, j):
            for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                if 0 <= x < m and 0 <= y < n:
                    yield x, y

        seen = set()
        def dfs(i, j, depth):
            if depth == len(word):
                return True

            for x, y in adj(i, j):
                if board[x][y] == word[depth] and (x, y) not in seen:
                    seen.add((x, y))  # mark seen until finishing
                    if dfs(x, y, depth + 1):
                        seen.discard((x, y))
                        return True
                    seen.discard((x, y))  # revert seen
            return False
        
        for i, rows in enumerate(board):
            for j, c in enumerate(rows):
                if c == word[0]:
                    seen.add((i, j))
                    if dfs(i, j, 1):
                        return True
                    seen.discard((i, j))
        return False
    
sol2 = Solution2()
```

</div>

worst case를 만들어서 성능테스트를 해보았다.  <br>
확인해보니, 두번째 버젼이 압도적으로 빠르게 끝났다. 

<div class="prompt input_prompt">
In&nbsp;[14]:
</div>

<div class="input_area" markdown="1">

```python
size = 40
board = random2Dcharacters(shape=(size, size), samples=['a', 'b'])
word = randomString(length=random.randint(0, size), samples=['a', 'b'])
print(np.array(board))
print(word)
print(sol1.exist(board, word, verbose=True))
print(sol2.exist(board, word, verbose=True))
```

</div>

{:.output_stream}

```
[['a' 'a' 'a' ... 'b' 'b' 'a']
 ['a' 'b' 'b' ... 'a' 'b' 'b']
 ['b' 'b' 'b' ... 'b' 'a' 'a']
 ...
 ['b' 'b' 'b' ... 'b' 'b' 'a']
 ['b' 'b' 'a' ... 'b' 'a' 'b']
 ['a' 'a' 'a' ... 'b' 'b' 'b']]
bbabbabbabaabbab
WorkingTime[exist]: 421.86666 ms
True
WorkingTime[exist]: 0.08345 ms
True

```
