---
title: "735.Asteroid Collision"
excerpt: "stack practice - how to use stack and how to think the use."
categories:
 - algorithms
tags:
 - incremental
 - stack
 - datastructure
use_math: true
last_modified_at: "2020-03-07"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/algorithms/algo.png
 overlay_filter: 0.5
 caption: how to think the use of stack when solving an algorithm?
 actions:
  - label: "leetcode"
  - url: "https://leetcode.com/problems/asteroid-collision/"
---
## Objective 
All elements of `ans` will be residual asteroids after all collision occur. <br>

## Intuition - Incremental Approach
> **`Flow of Incremental Approach`** <br>
  Assume that we have the result of given `~ (i-1)-th inputs`. <br>
  at `i-th iteration`, we should compute the result `~ i-th inputs` satisfying the objective of a problem. <br>
  In this way, we can get the final result by scanning `all inputs`.   


You can solve this problem by gradually scanning each asteroid and adding the result to `ans`, taking into account the crash cases. <br>
Assume that we already have `ans` when `i-th` iteration. <br>
When we meet `new `, which is a `i-th` asteroid, we can notice the fact as follows.
<div style="background-color: gray"> 
    <p>
        <span style="color:blue">Note that</span> asteroids of ans <b><span style="color:blue"> from the top to bottom </span></b> are affected until being stable when dealing with <span style="color: blue">i-th iteration</span> if collisions occur. 
    </p>
</div>

**Therefore**, we can use **`stack` data structure** to `ans`. <br>
We can use `stack` as `ans` by updating `ans` from top to bottom when dealing with `i-th iteration`, 

## How to solve using a stack in detail?
Append new asteroid for each iteration by considering as follows. <br>
If collision cases(`new < 0 < top`) occur, there are two cases.  
1. `abs(top) < abs(new)` - `ans` blows up. However, `new` is still alive. so, see `next top` of ans.  
2. `abs(top) == abs(new)` - both `ans` and `new` blow up. so, it becomes stable. (get out while loop) 

<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
class Solution(object):
    def asteroidCollision(self, asteroids):
        """
        :type asteroids: List[int]
        :rtype: List[int]
        """
        ans = [] # ans is a stack: elements of ans will be residual asteroids after all collision occur.
        for new in asteroids:
            # all collision cases
            # - At first, ans is not empty. (first requirement condition) 
            #   - Also, top of ans is postivie and new is negative.
            while ans and new < 0 < ans[-1]:
                # case 1 - top blows up, but new is still alive.
                if abs(ans[-1]) < abs(new):
                    ans.pop()
                    continue # see the next top of ans
                # case 2 - both top ans new blow up, it becomes stable.
                elif abs(ans[-1]) == abs(new):
                    ans.pop()
                # etc - ans does not change.
                break # get out the while loop.
            else: # [important]: to avoid the case that `new` blows up, but append the new to ans. 
                # if it goes through the while loop, where is above, avoid this line. 
                ans.append(new)
        return ans
sol = Solution()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
A = [5, 10, -5]
sol.asteroidCollision(A)
```

</div>




{:.output_data_text}

```
[5, 10]
```



## Report 
When I solved this problem, there were two things to think.
1. It was hard to think that `ans` is used as `stack` datastruture. <br>
    I cannot notice this fact when I solve this problem firstly. <br>
    However, I can figure out the use of stack makes the problem be solved incrementally.
2. Hard to implement the exception cases.
    * I could not know how to deal with collision cases.
        ```python
        while ans and new < 0 < ans[-1]:
        ```
        In python, at the while condition, 
        `ans` should be non-empty, and `new < 0 < ans[-1]`
        I thought that ans[-1] is impossible because when `ans` is empty at first step, the `-1` is out of index. 
        However, fortunately, In python, if first condition is false, the python does not see the second condition!.
        so, out of index error does not occur.
    * when `abs(ans[-1]) == abs(new):` occurs, both `ans` ans `new` blow up.
       Therefore, we should not append new because it disappears by collision. 
       I was confused at how to implement this.
       Luckly, In python, `while` or `else` is possible. 
       ```python
        while condition:
            # ...
        else: # [important]: to avoid the case that `new` blows up, but append the new to ans.  
            ans.append(new)
       ```
    

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
# test
while 1:
    print("hi")
    break
else:
    print("hello")
```

</div>

{:.output_stream}

```
hi

```

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
# test
while 0:
    print("hi")
    break
else:
    print("hello")

```

</div>

{:.output_stream}

```
hello

```
