---
title: "시험감독"
excerpt: "코딩 테스트 연습"
categories:
 - algorithms
tags:
 - calculate
use_math: true
last_modified_at: "2020-05-31"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/teaser.jpg
 overlay_filter: 0.5
---

# 시험 감독

[baekjoon](https://www.acmicpc.net/problem/13458)

너무 쉬운 문제.


<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
from sys import stdin
stdin = open('data/supervisor.txt')
input = stdin.readline
n = int(input())
a = list(map(int, input().split()))
b, c = map(int, input().split())

a = list(map(lambda x: x - b if x >= b else 0, a))
# print(a)
ans = n
for e in a:
    d, r = e // c, e % c
    if r: ans += d + 1
    else: ans += d
print(ans)
```

</div>

{:.output_stream}

```
10

```
