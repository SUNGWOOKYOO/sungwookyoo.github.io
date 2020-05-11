---
title: "241.Different Ways to Add Parentheses - leetcode"
excerpt: "Practice to utilize divide and conquer technique."
categories:
 - algorithms
tags:
 - DivideConquer
use_math: true
last_modified_at: "2020-05-11"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/teaser.jpg
 overlay_filter: 0.5
---

# 241. Different Ways to Add Parentheses

## Idea 
1. This algorithm divides `inp` string by operators. 
    * each operator is a dividing point, so reculsively call all possible cases.
    ```python
    for i in range(len(inp)):
        if not self.match(inp[i]):
            continue
        left = self.diffWaysToCompute(inp[:i])
        right = self.diffWaysToCompute(inp[i+1:])
    ```
2. From reculsive calling, If we get the `left` and `right` results, merge the results.
    * all possible combinataions are generated. 
    ```python
    for i in range(len(inp)):
        ...
        operator = inp[i]
        for l in left:
            for r in right:
                res.append(self.op(operator, l, r))
    ```
3. Base cases can be occured when `left` or `right` are empty.
    ```python
    if not bool(res):
        return [int(inp)]
    ```

<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
class Solution(object):
    def __init__(self):
        self.match = lambda c: True if c in ['+', '-', '*'] else False
    def op(self, o, a, b):
        if o == '+':
            return a + b
        elif o == '-':
            return a - b
        elif o == '*':
            return a * b
        assert o, 'invalid'
    def diffWaysToCompute(self, inp):
        """
        :type inp: str
        :rtype: List[int]
        """
        res = []
        for i in range(len(inp)):
            if not self.match(inp[i]):
                continue
            left = self.diffWaysToCompute(inp[:i])
            right = self.diffWaysToCompute(inp[i+1:])
            operator = inp[i]
            for l in left:
                for r in right:
                    res.append(self.op(operator, l, r))
        if not bool(res):
            return [int(inp)]
        return sorted(res)
```

</div>

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
inp = "2*3-4*5"
sol = Solution()
sol.diffWaysToCompute(inp)
```

</div>




{:.output_data_text}

```
[-34, -14, -10, -10, 10]
```



# referenece
[discuss](https://leetcode.com/problems/different-ways-to-add-parentheses/discuss/66328/A-recursive-Java-solution-(284-ms))
