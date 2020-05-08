---
title: "211.Add and Search Word - Leetcode"
excerpt: "Trie datastructure practice. "
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
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
from collections import deque, defaultdict
from treelib import Tree
from pprint import pprint
```

</div>

# Add and Search Word - Data structure design

Let $m$ be maximum length of a word, $n$ be the total number of words. <br>
Using the Trie datastructure, we can simply solve this problem in $O(mn)$.

## Implement 1 - use defaultdict

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
class WordDictionary(object):

    def __init__(self):
        """ Initialize your data structure here. """
        _root = lambda: defaultdict(_root)
        self.root = _root()
        self.isEnd = True

    def addWord(self, word):
        """ Adds a word into the data structure.
        :type word: str
        :rtype: None """
        cur = self.root
        for c in word:
            cur = cur[c]
        cur[self.isEnd] = word

    def search(self, word):
        """ Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        :type word: str
        :rtype: bool """

        def _search(cur: defaultdict, j: int = 0):
            if j == len(word):
                return cur != self.root and self.isEnd in cur
            c = word[j]
            if c == '.':
                res = False
                for k in cur.keys():
                    if isinstance(k, str):
                        res = res or _search(cur[k], j + 1)
                return res
            else:
                if c not in cur:
                    return False
                else:
                    return _search(cur[c], j + 1)

        return _search(cur=self.root, j=0)
```

</div>

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
# Your WordDictionary object will be instantiated and called as such:
# command = ["WordDictionary","addWord","addWord","addWord","addWord","addWord","addWord","addWord","addWord","search","search","search","search","search","search","search","search","search","search"]
# words = [[],["ran"],["rune"],["runner"],["runs"],["add"],["adds"],["adder"],["addee"],["r.n"],["ru.n.e"],["add"],["add."],["adde."],[".an."],["...s"],["....e."],["......."],["..n.r"]]
command = ["WordDictionary", "addWord", "addWord", "addWord", "search", "search", "search", "search"]
words = [[], ["bad"], ["dad"], ["mad"], ["pad"], ["bad"], [".ad"], ["b.."]]
obj = WordDictionary()
cmds = {'addWord': obj.addWord, 'search': obj.search}
ans = []
for cmd, w in list(zip(command, words))[1:]:
    res = cmds[cmd](w[0])
    if cmd == 'search':
        ans.append(res)

pprint(obj.root)
print(ans)
```

</div>

{:.output_stream}

```
defaultdict(<function WordDictionary.__init__.<locals>.<lambda> at 0x7f04ec0dd200>,
            {'b': defaultdict(<function WordDictionary.__init__.<locals>.<lambda> at 0x7f04ec0dd200>,
                              {'a': defaultdict(<function WordDictionary.__init__.<locals>.<lambda> at 0x7f04ec0dd200>,
                                                {'d': defaultdict(<function WordDictionary.__init__.<locals>.<lambda> at 0x7f04ec0dd200>,
                                                                  {True: 'bad'})})}),
             'd': defaultdict(<function WordDictionary.__init__.<locals>.<lambda> at 0x7f04ec0dd200>,
                              {'a': defaultdict(<function WordDictionary.__init__.<locals>.<lambda> at 0x7f04ec0dd200>,
                                                {'d': defaultdict(<function WordDictionary.__init__.<locals>.<lambda> at 0x7f04ec0dd200>,
                                                                  {True: 'dad'})})}),
             'm': defaultdict(<function WordDictionary.__init__.<locals>.<lambda> at 0x7f04ec0dd200>,
                              {'a': defaultdict(<function WordDictionary.__init__.<locals>.<lambda> at 0x7f04ec0dd200>,
                                                {'d': defaultdict(<function WordDictionary.__init__.<locals>.<lambda> at 0x7f04ec0dd200>,
                                                                  {True: 'mad'})})})})
[False, True, True, True]

```

## Implement 2 - Node 

This is a normal way to implement. <br>
For easy understanding this solution, I visualize trie data structure.

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
class Node:
    def __init__(self, identifier):
        self.identifier = identifier
        self.children = {}
        self.isEnd = False


class WordDictionary(object):

    def __init__(self):
        """ Initialize your data structure here. """
        self.root = Node('root')

    def addWord(self, word):
        """ Adds a word into the data structure.
        :type word: str
        :rtype: None """
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = Node(c)
            cur = cur.children[c]
        cur.isEnd = True

    def search(self, word):
        """ Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        :type word: str
        :rtype: bool """

        def _search(cur: Node, j: int = 0):
            if j == len(word):
                return cur != self.root and cur.isEnd
            c = word[j]
            if c == '.':
                res = False
                for k in cur.children:
                    res = res or _search(cur.children[k], j + 1)
                return res
            else:
                if c not in cur.children:
                    return False
                else:
                    return _search(cur.children[c], j + 1)

        return _search(cur=self.root, j=0)

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
In&nbsp;[7]:
</div>

<div class="input_area" markdown="1">

```python
# Your WordDictionary object will be instantiated and called as such:
command = ["WordDictionary","addWord","addWord","addWord","addWord","addWord","addWord","addWord","addWord","search","search","search","search","search","search","search","search","search","search"]
words = [[],["ran"],["rune"],["runner"],["runs"],["add"],["adds"],["adder"],["addee"],["r.n"],["ru.n.e"],["add"],["add."],["adde."],[".an."],["...s"],["....e."],["......."],["..n.r"]]
# command = ["WordDictionary", "addWord", "addWord", "addWord", "search", "search", "search", "search"]
# words = [[], ["bad"], ["dad"], ["mad"], ["pad"], ["bad"], [".ad"], ["b.."]]
obj = WordDictionary()
cmds = {'addWord': obj.addWord, 'search': obj.search}
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
├── a
│   └── d
│       └── d
│           ├── e
│           │   ├── e
│           │   └── r
│           └── s
└── r
    ├── a
    │   └── n
    └── u
        └── n
            ├── e
            ├── n
            │   └── e
            │       └── r
            └── s

[('r.n', True), ('ru.n.e', False), ('add', True), ('add.', True), ('adde.', True), ('.an.', False), ('...s', True), ('....e.', True), ('.......', False), ('..n.r', False)]
[True, False, True, True, True, False, True, True, False, False]

```
