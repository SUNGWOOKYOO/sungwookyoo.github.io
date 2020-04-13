---
title: "1023. Camelcase Matching Leetcode using python"
excerpt: "Trie datastructure practice"
categories:
 - algorithms
tags:
 - datastructure
use_math: true
last_modified_at: "2020-04-13"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
overlay_image: /assets/images/algorithms/algo.png
 overlay_filter: 0.5
---

<div class="prompt input_prompt">
In&nbsp;[None]:
</div>

<div class="input_area" markdown="1">

```python
from collections import defaultdict
from pprint import pprint
```

</div>

# 1023. Camelcase Matching

[leetcode](https://leetcode.com/problems/camelcase-matching/)

$n$ queries, where the maximum query length is $m$

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
queries = ["FooBar","FooBarTest","FootBall","FrameBuffer","ForceFeedBack"]
pattern = "FoBaT"
```

</div>

## Naive

### Key Idea
* Uppercase letters fit in order with given a pattern.
* Lowercase and uppercase letters match the given pattern.

#### Skill
```python
all(c in it for c in s)
```

Check Subsequence
$$
O(nm)
$$

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
def camelMatch(qs, p):
    def u(s):  
        """ Uppercase letters fit in order with a given pattern. """
        return [c for c in s if c.isupper()]

    def issup(s, t):
        """ Lowercase and uppercase letters match the given pattern. """
        it = iter(t)
        # return all(c in it for c in s)
        checks = []
        for c in s:
            ck = (c in it, c)
            # print(ck)
            checks.append(ck[0])
        # print(checks)
        return all(checks)
            
    return [u(p) == u(q) and issup(p, q) for q in qs]
```

</div>

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
q = ["FrameBuffer"]
p = "FoBa"
camelMatch(q, p)
```

</div>




{:.output_data_text}

```
[False]
```



## Trie

### Key Idea
* Build a `trie` data structure with given a `pattern`.
* Check every query to determine `True` of `False`.

#### checking process. 
check every character of a query(by following the trie that was built before), where the character for each iteration is `c`
* `c` is uppercase, but has no child -> `False`
* `c` in current node's children(c can be uppercase or lowercase) -> see next node in Trie.
* `c` is lowercase, but has no child -> ignore it and continue(it is not necessarily to implement.)
* When terminate checking process, return `True` if current node's has `isEnd` flag else `False`

#### Time complexity analysis
$$
O(nm)
$$

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
class Solution(object):
    def camelMatch(self, queries, pattern):
        """
        :type queries: List[str]
        :type pattern: str
        :rtype: List[bool]
        """
        _root = lambda: defaultdict(_root)
        root = _root()
        cur = root
        isEnd = True
        for c in pattern:
            cur = cur[c]
        cur[isEnd] = pattern
        pprint(root)

        def check(word):
            cur = root
            for c in word:
                if c.isupper() and c not in cur:
                    return False
                elif c in cur:
                    cur = cur[c]
            return isEnd in cur

        return list(map(check, queries))

sol = Solution()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
sol.camelMatch(queries, pattern)
```

</div>

{:.output_stream}

```
defaultdict(<function Solution.camelMatch.<locals>.<lambda> at 0x7fb3744dcb90>,
            {'F': defaultdict(<function Solution.camelMatch.<locals>.<lambda> at 0x7fb3744dcb90>,
                              {'o': defaultdict(<function Solution.camelMatch.<locals>.<lambda> at 0x7fb3744dcb90>,
                                                {'B': defaultdict(<function Solution.camelMatch.<locals>.<lambda> at 0x7fb3744dcb90>,
                                                                  {'a': defaultdict(<function Solution.camelMatch.<locals>.<lambda> at 0x7fb3744dcb90>,
                                                                                    {'T': defaultdict(<function Solution.camelMatch.<locals>.<lambda> at 0x7fb3744dcb90>,
                                                                                                      {True: 'FoBaT'})})})})})})

```




{:.output_data_text}

```
[False, True, False, False, False]
```



## Regex 

[How to use Regular Expression?](https://wikidocs.net/4308#match)

<div class="prompt input_prompt">
In&nbsp;[7]:
</div>

<div class="input_area" markdown="1">

```python
import re
```

</div>

given a pattern, build regular expression as follows.

<div class="prompt input_prompt">
In&nbsp;[8]:
</div>

<div class="input_area" markdown="1">

```python
q = ["FooBarTest"]
p = "FoBaT"
print("^[a-z]*" + "[a-z]*".join(p) + "[a-z]*$", p)
re.match("^[a-z]*" + "[a-z]*".join(p) + "[a-z]*$", q[0]) # return matched object if match else Nones
```

</div>

{:.output_stream}

```
^[a-z]*F[a-z]*o[a-z]*B[a-z]*a[a-z]*T[a-z]*$ FoBaT

```




{:.output_data_text}

```
<re.Match object; span=(0, 10), match='FooBarTest'>
```



### Key Idea

If q is matched with Regex, there is an `re.object` <br>
Else, return `None` <br>
Therefore, determine True or False by existence of `re.object` after matching process.

<div class="prompt input_prompt">
In&nbsp;[9]:
</div>

<div class="input_area" markdown="1">

```python
def camelMatch(qs, p):
    return [re.match("^[a-z]*" + "[a-z]*".join(p) + "[a-z]*$", q) != None for q in qs]
```

</div>

<div class="prompt input_prompt">
In&nbsp;[10]:
</div>

<div class="input_area" markdown="1">

```python
camelMatch(q, p)
```

</div>




{:.output_data_text}

```
[True]
```


