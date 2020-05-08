---
title: "676.Magic Dictionary"
excerpt: "Trie data structure practice "
categories:
 - algorithms
tags:
 - datastructure
use_math: true
last_modified_at: "2020-05-08"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/algorithms/algo.png
 overlay_filter: 0.5
---

<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
from typing import List
from collections import defaultdict, deque, Counter
from treelib import Tree
from pprint import pprint
```

</div>

# 676. Implement Magic Dictionary

## Objective:
>Given $n$ words, where words' maximum length be $m$. <br>
Build words dictionary, and then check if there exist matching word that **except for only one character**.

## Idea
build trie and use dfs search for checking process. <br>
`cnt`: allow only one different character in the given query word. <br>
when `cnt = 0`, search all characters by setting `cnt = 1`. <br>
else, the process is same with general trie search algorithm.
> If you wonder general trie algorithm, please visit [this document](https://sungwookyoo.github.io/algorithms/Trie/)

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
class Node:
    def __init__(self, identifier):
        self.identifier = identifier
        self.children = {}
        self.isEnd = False


class MagicDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = Node('root')

    def buildDict(self, words: List[str]) -> None:
        """
        Build a dictionary through a list of words
        """
        for word in words:
            cur = self.root
            for c in word:
                if c not in cur.children:
                    cur.children[c] = Node(c)
                cur = cur.children[c]
            cur.isEnd = True

    def search(self, word: str) -> bool:
        """
        Returns if there is any word in the trie that equals to the given word after modifying exactly one character
        """

        def _search(cur: Node, j: int = 0, cnt: int = 0):
            if j == len(word):
                return cur != self.root and cur.isEnd and cnt == 1
            c = word[j]
            if cnt == 0:
                res = False
                for k in cur.children:
                    res = res or _search(cur.children[k], j + 1, cnt=1 if c != k else 0)
                return res
            else:
                if c not in cur.children:
                    return False
                return _search(cur.children[c], j + 1, cnt=cnt)
        return _search(cur=self.root, j=0, cnt=0)

    def show(self):
        s = self.root
        queue = deque([s])
        tree = Tree()
        tree.create_node(tag='root', identifier=s)
        while queue:
            u = queue.popleft()
            for v in u.children.values():
                queue.append(v)
                tree.create_node(tag=v.identifier,
                                 identifier=v,
                                 parent=u)
        tree.show()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
command = ["MagicDictionary", "buildDict", "search", "search", "search", "search"]
words = [[], [["hello","hallo","leetcode"]], ["hello"], ["hhllo"], ["hell"], ["leetcoded"]]

obj = MagicDictionary()
cmds = {'buildDict': obj.buildDict, 'search': obj.search}
queries, ans = [], []
for cmd, w in list(zip(command, words))[1:]:
    res = cmds[cmd](w[0])
    if cmd == 'search':
        queries.append(w[0])
        ans.append(res)

obj.show()
print(list(zip(queries, ans)))
print(ans)
```

</div>

{:.output_stream}

```
root
├── h
│   ├── a
│   │   └── l
│   │       └── l
│   │           └── o
│   └── e
│       └── l
│           └── l
│               └── o
└── l
    └── e
        └── e
            └── t
                └── c
                    └── o
                        └── d
                            └── e

[('hello', True), ('hhllo', True), ('hell', False), ('leetcoded', False)]
[True, True, False, False]

```

## Discuss

[another solution](https://leetcode.com/problems/implement-magic-dictionary/discuss/107454/Python-without-*26-factor-in-complexity) is possible 

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
class MagicDictionary(object):
    def _candidates(self, word):
        for i in range(len(word)):
            yield word[:i] + '*' + word[i + 1:]

    def buildDict(self, words):
        self.words = set(words)
        self.near = Counter(cand for word in words
                                        for cand in self._candidates(word))

    def search(self, word):
        return any(self.near[cand] > 1 or
                   self.near[cand] == 1 and word not in self.words
                   for cand in self._candidates(word))
```

</div>

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
command = ["MagicDictionary", "buildDict", "search", "search", "search", "search"]
words = [[], [["hello","hallo","leetcode"]], ["hello"], ["hhllo"], ["hell"], ["leetcoded"]]

obj = MagicDictionary()
cmds = {'buildDict': obj.buildDict, 'search': obj.search}
ans = []
for cmd, w in list(zip(command, words))[1:]:
    res = cmds[cmd](w[0])
    if cmd == 'search':
        ans.append(res)

print(obj.words)
print(obj.near)
print(ans)
```

</div>

{:.output_stream}

```
{'hello', 'hallo', 'leetcode'}
Counter({'h*llo': 2, '*ello': 1, 'he*lo': 1, 'hel*o': 1, 'hell*': 1, '*allo': 1, 'ha*lo': 1, 'hal*o': 1, 'hall*': 1, '*eetcode': 1, 'l*etcode': 1, 'le*tcode': 1, 'lee*code': 1, 'leet*ode': 1, 'leetc*de': 1, 'leetco*e': 1, 'leetcod*': 1})
[True, True, False, False]

```
