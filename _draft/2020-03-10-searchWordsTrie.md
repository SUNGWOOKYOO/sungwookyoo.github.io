---
title: "#"
excerpt: "#"
categories:
 - #
tags:
 - #
use_math: true
last_modified_at: "2020-03-10"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/teaser.jpg
 overlay_filter: 0.5
 caption: #
 actions:
  - label: "#"
    url: "#"
---

# [2020 카카오 블라인드 공채] 가사 검색

[YouTube](https://raon0229.tistory.com/64) 를 바탕으로 코딩하여 풀었다. 
Tri 자료구조에 대해 배울 수 있었고, 이를 응용한 방법을 코딩해 보았다.

what is [Trie](https://www.geeksforgeeks.org/trie-insert-and-search/)?

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
import sys

class TrieNode:
    # Trie node class
    def __init__(self):
        self.children = {}
        # isEndOfWord is True if node represent the end of the word
        # self.isEndOfWord = False # Not used in this problem.
        self.info = {} # {level(from node): # of children}
class Trie:
    # Trie data structure class
    def __init__(self):
        self.root = self.getNode()

    def getNode(self):
        """ returns new trie node. (initialized to NULLs)  """
        return TrieNode()

    def insert(self, key):
        """ insert a string to trie
        Args:
            key: a word
        """
        cur = self.root
        length = len(key)
        # update cur info: {lv: # of children}
        cur.info[length] = cur.info.get(length, 0) + 1
        for c in key:
            if c not in cur.children:
                cur.children[c] = self.getNode()
            cur = cur.children[c]
            # update next cur
            length -= 1
            cur.info[length] = cur.info.get(length, 0) + 1

        # mark last node as leaf
        cur.isEndOfWord = True

    def search(self, key, wlen):
        """ return # of matching children, given key and wlen
        e.g., Assume that given key="fro", wlen="??"
                Also, there are "frodo", "front", "frozen" in trie.
                return 2.
        Args:
            key: string to find(substring query except for '?')
            wlen: # of wild card characters
        """
        cur = self.root
        if len(key) + wlen not in cur.info:
            # if total length=len(key)+wlen is not in cur.info,
            # there are no words to match.
            return 0

        # go into the node with key[-1]
        for c in key:
            if c not in cur.children:
                return 0
            cur = cur.children[c]

        return cur.info.get(wlen, 0)

def solution(words, queries):
    t = Trie()
    inv_t = Trie()
    for w in words:
        t.insert(w)
        inv_t.insert(w[::-1])

    ans = []
    for q in queries:
        sub = q[:q.find('?')] if q[0] != "?" else q[::-1][:q[::-1].find('?')]
        wlen = len(q) - len(sub)
        if q[0] == '?':
            ans.append(inv_t.search(sub, wlen))
        else:
            ans.append(t.search(sub, wlen))
    return ans

def main():
    sys.stdin = open("input.txt", "r")
    words = list(map(str, sys.stdin.readline().split()))
    queries = list(map(str, sys.stdin.readline().split()))
    # ans = naive(words, queries)
    ans = solution(words, queries)
    print(ans)

if __name__ == "__main__":
    main()
```

</div>

{:.output_stream}

```
[3, 2, 4, 1, 0]

```
