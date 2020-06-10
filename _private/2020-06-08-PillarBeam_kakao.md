---
title: "기둥과 보 설치"
excerpt: "시뮬레이션 능력 점검"
categories:
 - algorithms
tags:
 - enumerate
 - simulation
 - kakao
use_math: true
last_modified_at: "2020-06-08"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/teaser.jpg
 overlay_filter: 0.5
---

# 기둥과 보 설치 - 카카오 공채 문제
[programmers](https://programmers.co.kr/learn/courses/30/lessons/60061)

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
import sys
sys.path.append("/home/swyoo/algorithm/")
from utils.verbose import logging_time, printProgressBar
```

</div>

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
def solution(n, build_frame):
    def check(states):
        for x, y, what in states:
            if what == 0:
                # 바닥이거나 양쪽 중 한쪽에 보가 있거나 바로 밑에 기둥이 있어야함.
                if not (y == 0 or (x - 1, y, 1) in states or (x, y, 1) in states or (x, y - 1, 0) in states):
                    return False
            elif what == 1:
                # 양쪽 중 한 쪽에 기둥이 있거나, 양쪽 모두에 보가 있어야 함.
                if not ((x, y - 1, 0) in states or (x + 1, y - 1, 0) in states or ((x - 1, y, 1) in states and (x + 1, y, 1) in states)):
                    return False
        return True  # all pass

    """ build_frame 의 하나의 명령이 주어졌을때,  
    설치 명령 인경우,
        수행해본 뒤 state를 봐가면서 조건을 만족하는지 보고, 조건에 맞지않다면 state를 roll back.
    삭제 명령인 경우, 
        삭제해본뒤  state를 봐가면서 조건을 만족하는지 보고, 조건에 맞지않다면 state를 roll back.
    build_frame안에 m 개의 명령이 있다면, 체크하는데 최대 m 시간이 걸릴것이다, 따라서 O(m^2), 마지막 sort는 mlogm. """
    states = set()  # avoid duplication.
    for exec in build_frame:
        x, y, what, how = exec
        if how == 1:  # 설치
            states.add((x, y, what))
            if not check(states):
                states.remove((x, y, what))
        elif how == 0:  # 삭제
            states.remove((x, y, what))
            if not check(states):
                states.add((x, y, what))
    return sorted(list(map(list, states)), key=lambda x: (x[0], x[1], x[2]))
```

</div>

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
n, build_frame =  5, [[0, 0, 0, 1], [2, 0, 0, 1], [4, 0, 0, 1], [0, 1, 1, 1], [1, 1, 1, 1], [2, 1, 1, 1], [3, 1, 1, 1], [2, 0, 0, 0], [1, 1, 1, 0], [2, 2, 0, 1]], 
gt = [[0, 0, 0], [0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1], [4, 0, 0]]
print(n, build_frame, gt)
assert gt == solution(n, build_frame)
```

</div>

{:.output_stream}

```
5 [[0, 0, 0, 1], [2, 0, 0, 1], [4, 0, 0, 1], [0, 1, 1, 1], [1, 1, 1, 1], [2, 1, 1, 1], [3, 1, 1, 1], [2, 0, 0, 0], [1, 1, 1, 0], [2, 2, 0, 1]] [[0, 0, 0], [0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1], [4, 0, 0]]

```

# Report 

매우 귀찮은 일이지만, 주어진 조건이 맞는지 검사하는 코드를 짜는 것도 하나의 능력으로 테스트 된다.

* 없는 것을 지우거나, 설치한 것을 또 설치하는 입력을 주지는 않는 다고 문제에서 언급했으므로 굳이 체크할 필요는 없다. <br>
[referenced korean blog](https://m.post.naver.com/viewer/postView.nhn?volumeNo=26959882&memberNo=33264526)

* skill-up
```python
 sorted(list(map(list, states)), key=lambda x: (x[0], x[1], x[2]))
```
이렇게하면 원하는 순대로 정렬이 되니 손 쉽게 원하는 결과를 얻을 수 있다.
