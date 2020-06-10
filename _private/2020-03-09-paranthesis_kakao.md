---
title: "괄호변환 - 카카오 공채 문제풀이"
excerpt: "카카오 공채 코딩 문제풀이"
categories:
 - algorithms
tags:
 - stack
 - DivideConquer
 - kakao 
use_math: true
last_modified_at: "2020-03-09"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/teaser.jpg
 overlay_filter: 0.5
 caption: kakao
 actions:
  - label: "kakao"
    url: "https://tech.kakao.com/2019/10/02/kakao-blind-recruitment-2020-round1/"
---

# [2020카카오공채] 괄호 변환

[programmers](https://programmers.co.kr/learn/courses/30/lessons/60058)

1. 입력이 빈 문자열인 경우, 빈 문자열을 반환합니다.
2. 문자열 w를 두 "균형잡힌 괄호 문자열" u, v로 분리합니다. 단, u는 "균형잡힌 괄호 문자열"로 더 이상 분리할 수 없어야 하며, v는 빈 문자열이 될 수 있습니다.
3. 문자열 u가 "올바른 괄호 문자열" 이라면 문자열 v에 대해 1단계부터 다시 수행합니다.
  3-1. 수행한 결과 문자열을 u에 이어 붙인 후 반환합니다.
4. 문자열 u가 "올바른 괄호 문자열"이 아니라면 아래 과정을 수행합니다.
  4-1. 빈 문자열에 첫 번째 문자로 '('를 붙입니다.
  4-2. 문자열 v에 대해 1단계부터 재귀적으로 수행한 결과 문자열을 이어 붙입니다.
  4-3. ')'를 다시 붙입니다.
  4-4. u의 첫 번째와 마지막 문자를 제거하고, 나머지 문자열의 괄호 방향을 뒤집어서 뒤에 붙입니다.
  4-5. 생성된 문자열을 반환합니다.

중요한 가정: 항상 `p`의 '(' 와 ')' 수는 동일하다.
[REF](https://m.post.naver.com/viewer/postView.nhn?volumeNo=26897018&memberNo=33264526)

<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
def solution(p, verbose=False):
    """ fix parenthesis """
    def is_balance(x):
        """ return an index of balanced point."""
        bal = 0
        for k, e in enumerate(x):
            if e == '(':
                bal += 1
            else:
                bal -= 1
            if bal == 0:
                break
        return k
    
    def is_right(x):
        """ check if x is right
        1. if e == '(', push to stack.
        2. elif e == ')'
             * if '(' in stack, that is non-empty, pop the top element.
             * elif stack is empty, return False because it is not correct.
        """
        ck = []
        for e in x:
            if e == '(':
                ck.append(e)
            else:
                if ck != []: ck.pop()
                else: return False
        return True if ck == [] else False
    
    def reverse(x):
        rev = ''
        for e in x:
            if e == '(':
                rev += ')'
            else:
                rev += '('
        return rev
    
    if p == '': return p
    k = is_balance(p)
    u, v = p[:k+1], p[k+1:]
    if verbose: print("u= {}, v= {}".format(u, v))
    
    if is_right(u):
        if verbose: print('right')
        return u + solution(v)
    left = '(' + solution(v) + ')'
    right = reverse(u[1:-1])
    return left + right                
```

</div>

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
p = "()))((()"
solution(p, verbose=False)
```

</div>




{:.output_data_text}

```
'()(())()'
```



## Report
문제에서 주어진대로 정확히 구현하면 통과 가능.
