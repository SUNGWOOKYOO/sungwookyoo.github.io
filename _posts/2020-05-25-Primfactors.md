---
title: "Find Prime Factors"
excerpt: "how to find prim factors"
categories:
 - algorithms
tags:
 - calculate
use_math: true
last_modified_at: "2020-05-25"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/teaser.jpg
 overlay_filter: 0.5
---

<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
from collections import Counter
from functools import reduce
```

</div>

## Prime Factors
<details> <summary> Prime Factor </summary>
<p> Prime factor is the factor of the given number which is a prime number. <br>
    Factors are the numbers you multiply together to get another number. <br>
    In simple words, prime factor is finding which prime numbers multiply together to make the original number. </p>
<img src="https://media.geeksforgeeks.org/wp-content/uploads/6-min-1.png" width="600">
</details>

<div class="prompt input_prompt">
In&nbsp;[16]:
</div>

<div class="input_area" markdown="1">

```python
def prime_factors(n):
    """ A function to print all prime factors of a given number n """
    i = 2
    factors = []
    while i * i <= n:
        if n % i:  # if there is a remainder.
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors
```

</div>

## Find the number of divisors

If $10 = 2^2 \times 5^2$, the number of divisors as follows:
$(2 + 1) \times (2 + 1) = 9$ 

<div class="prompt input_prompt">
In&nbsp;[22]:
</div>

<div class="input_area" markdown="1">

```python
def numOfDivs(n, verbose=False):
    primes =  Counter(prime_factors(n))
    if verbose: print(primes)
    cnts = list(map(lambda x : x + 1, primes.values()))
    return reduce(lambda x, y: x * y, cnts)
```

</div>

<div class="prompt input_prompt">
In&nbsp;[23]:
</div>

<div class="input_area" markdown="1">

```python
numOfDivs(100, verbose=True)
```

</div>

{:.output_stream}

```
Counter({2: 2, 5: 2})

```




{:.output_data_text}

```
9
```


