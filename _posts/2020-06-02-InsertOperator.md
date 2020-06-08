---
title: "연산자 끼워넣기"
excerpt: "DFS practice"
categories:
 - algorithms
tags:
 - DFS
 - enumerate
 - samsung
use_math: true
last_modified_at: "2020-06-02"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/teaser.jpg
 overlay_filter: 0.5
---

# 연산자 끼워넣기

모든 수 $n$개, 연산자수 $n - 1$개 이다. <br>
주어진 연산자의 counter 수에 따라 경우의 수가 달라진다. <br>
에를 들면, <br>
주어진 수는 `1, 2, 3, 4, 5, 6`이고, <br>
주어진 연산자가 `덧셈(+) 2개, 뺄셈(-) 1개, 곱셈(×) 1개, 나눗셈(÷) 1개`인 경우 <br>
${n \choose 2} \times 3! = 60$가지의 경우의 수가 나온다.

모든 경우의 수에 대해 call해보고, 그 과정에서 $max, min$ 값을 update하자. <br>

**핵심**은 각각의 **연산자 숫자**와 **현재 local 값**을 인자로 받아야 모든 cases를 추적할 수 있다. 

## Submitted Code

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
from sys import stdin
stdin = open('data/InsertOperators.txt')

input = stdin.readline
n = int(input())
a = list(map(int, input().split()))
nums = list(map(int, input().split()))
cnts = [('+', nums[0]), ('-', nums[1]), ('*', nums[2]), ('/', nums[3])]

def solution(a, cnts):
    INF = 1e20
    maxv, minv = -INF, INF
    def naive(i, cnts, loc):
        nonlocal maxv, minv

        if i == len(a):
            maxv, minv = max(maxv, loc), min(minv, loc)

        for op, cnt in cnts:
            if not cnt: continue
            if op == '+':
                naive(i + 1, [('+', cnt - 1)] + cnts[1:], loc + a[i])
            elif op == '-':
                naive(i + 1, cnts[:1] + [('-', cnt - 1)] + cnts[2:], loc - a[i])
            elif op == '*':
                naive(i + 1, cnts[:2] + [('*', cnt - 1)] + cnts[3:], loc * a[i])
            elif op == '/':
                if loc >= 0: naive(i + 1, cnts[:-1] + [('/', cnt - 1)], loc // a[i])
                else: naive(i + 1, cnts[:-1] + [('/', cnt - 1)], -(-loc // a[i]))

    naive(1, cnts, a[0])
    return maxv, minv

maxv, minv = solution(a, cnts)
print(maxv)
print(minv)
```

</div>

{:.output_stream}

```
54
-24

```

## 비슷한 문제들

1. [9480. 민정이와 광직이의 알파벳 공부 - Beakjoon ](https://sungwookyoo.github.io/algorithms/AlphabetStudy/)
2. [494. Target Sum - Leetcode](https://sungwookyoo.github.io/algorithms/TargetSum/)
