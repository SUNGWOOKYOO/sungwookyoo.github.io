---
title: "169.Majority Element - Leetcode"
excerpt: "Find Majority Elment in a given array"
categories:
 - algorithms
tags:
 - DivideConquer
 - datastructure
use_math: true
last_modified_at: "2020-03-24"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/algorithms/algo.png
 overlay_filter: 0.5
 caption: algorithm
 actions:
  - label: "Leetcode"
    url: "https://leetcode.com/problems/majority-element/"
---

<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
import sys
from collections import Counter, defaultdict
sys.path.append("/home/swyoo/algorithm")
from utils.verbose import logging_time
```

</div>

# 169. Majority Element
[leetcode](https://leetcode.com/problems/majority-element/)

Given an array of size n, find the majority element. The majority element is the element that appears more than $⌊ n/2 ⌋$ times. <br>
You may assume that **the array is non-empty** and **the majority element always exist** in the array.

## Divide and Conquer
majority element가 항상 존재한다는 것에 주목하면, `left`또는`right`는 그 범위 안에서 항상 정답이다. <br>
또한, majority element는 **$⌊ n/2 ⌋$ 이상 발생한 숫자**이므로, cross하는 경우에 대해서 `left`아니면 `right`가 답이 된다. <br>
majority element는 count를 통해 알 수 있다. <br>
`left`(`nums[s..mid]`의 majaority element), `right`(`nums[mid+1..e]`의 majaority element)를 recursive하게 구해서 안다는 가정하에 <br>
`nums[s..e]`의 majaority element)를 다음과 같이 구한다. <br>
* `left`와 `right`가 같다면 그대로 `left`또는 `right`를 majority로 한다. 
* `left`와 `right`가 다르다면 `left` 와 `right` 둘 중 하나가 majority이므로 count해보고 결정한다. 

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
class Solution(object):
    # O(nlogn) - divide and conquer 
    @logging_time
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        def dc(s, e):
            if s >= e: return nums[s]
            
            mid = (s + e)//2
            left = dc(s, mid)
            right = dc(mid + 1, e)
            
            # cross
            if left == right:
                return left
            count_left = sum(1 for k in nums[s: e + 1] if k == left)
            count_right = sum(1 for k in nums[s: e + 1] if k == right)
            return left if count_left > count_right else right
        
        return dc(0, len(nums)-1)
    
    # O(n) - use hashmap
    @logging_time
    def majorityElement_v2(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
#         def f(x, H):
#             H[x] += 1
#         H = defaultdict(int)
#         map(lambda e: f(e, H), nums)
#         return max(H.keys(), key=lambda x: H[x])
        counts = Counter(nums)
        return max(counts.keys(), key=counts.get)
```

</div>

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
sol = Solution()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
A = [2,2,1,1,1,2,2]
print(sol.majorityElement(A, verbose=True))
print(sol.majorityElement_v2(A, verbose=True))
```

</div>

{:.output_stream}

```
WorkingTime[majorityElement]: 0.01216 ms
2
WorkingTime[majorityElement_v2]: 0.07081 ms
2

```
