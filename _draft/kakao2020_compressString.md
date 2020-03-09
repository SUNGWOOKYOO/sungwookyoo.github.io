<!-->
---
title: "kakao2020_compressString"
excerpt: "Need_modify"
categories:
 - Need_modify
tags:
 - Need_modify
last_modified_at: 2020-03-00
use_math: true
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header: 
 overlay_image: 
 overlay_filter: 0.5
 caption: Need_modify
 actions
  - label: "Need_modify"
  - url: "Need_modify"
---
<-->
# 문자열 압축

<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
def solution(s, verbose=False):
    
    N = len(s)
    ans = 1e8
    for slen in range(1, N//2 + 1):    
        seen = False
        cnt = slen
        for i in range(0, N-slen, slen):
            sub = s[i: i+slen]
            tmp = s[i+slen: i+2*slen]
            if verbose: print("sub:{:<4}| tmp:{:>4}".format(sub, tmp), end="")
            if sub == tmp:
                if not seen: 
                    seen = True
                    cnt += 1
                    if verbose: print("\t> +1, {}".format(cnt))
                elif verbose: print("seen '{}' before".format(tmp))
            else:
                cnt += len(tmp) # note that the length of tmp is not always slen
                seen = False # reset seen flag
                if verbose: print("\t> +{}, {}".format(slen, cnt))
        ans = min(ans, cnt)
    
    return ans
```

</div>

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
string = "abcabcdede"
```

</div>

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
solution(string, verbose=True)
```

</div>

{:.output_stream}

```
sub:a   | tmp:   b	> +1, 2
sub:b   | tmp:   c	> +1, 3
sub:c   | tmp:   a	> +1, 4
sub:a   | tmp:   b	> +1, 5
sub:b   | tmp:   c	> +1, 6
sub:c   | tmp:   d	> +1, 7
sub:d   | tmp:   e	> +1, 8
sub:e   | tmp:   d	> +1, 9
sub:d   | tmp:   e	> +1, 10
sub:ab  | tmp:  ca	> +2, 4
sub:ca  | tmp:  bc	> +2, 6
sub:bc  | tmp:  de	> +2, 8
sub:de  | tmp:  de	> +1, 9
sub:abc | tmp: abc	> +1, 4
sub:abc | tmp: ded	> +3, 7
sub:ded | tmp:   e	> +3, 8
sub:abca| tmp:bcde	> +4, 8
sub:bcde| tmp:  de	> +4, 10
sub:abcab| tmp:cdede	> +5, 10

```




{:.output_data_text}

```
8
```


