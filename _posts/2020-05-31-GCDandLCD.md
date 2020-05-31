---
title: "GCD 와 LCD 구하기"
excerpt: "최대공약수, 최소공배수를 구해라"
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

# GCD

According to [Euclidean Algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm), <br>
GCD(Greatest Common Divors) of `a, b` is same with GCD of `b, a % b`

Recursive Approach as follows.

<div class="prompt input_prompt">
In&nbsp;[18]:
</div>

<div class="input_area" markdown="1">

```python
def gcd(a, b):
    return a if b == 0 else gcd(b, a % b)
gcd(60, 48)
```

</div>




{:.output_data_text}

```
12
```



Iterative Approach as follows.

<div class="prompt input_prompt">
In&nbsp;[19]:
</div>

<div class="input_area" markdown="1">

```python
def gcd(a, b):
    while b:
        a, b = b, a%b
    return a
gcd(60, 48)
```

</div>




{:.output_data_text}

```
12
```



# LCD

LCD is easy if you understand GCD. <br>
Simply divide `a, b` by `gcd(a, b)`.

<div class="prompt input_prompt">
In&nbsp;[24]:
</div>

<div class="input_area" markdown="1">

```python
def lcd(a, b):
    return int(a * b / gcd(a, b))

lcd(60, 48)
```

</div>




{:.output_data_text}

```
240
```



## Reference

[1] [Korean Blog](https://astrohsy.tistory.com/4) <br>
[2] [Euclidean Algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm) <br>
