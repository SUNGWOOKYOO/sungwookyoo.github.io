---
title: "logging time using decorator in python"
excerpt: "this post describes how to use decorator for logging time in python"

categories:
  - tips
tags:
  - python
use_math: true
last_modified_at: 2020-02-29
toc: true
toc_sticky: true
toc_label: "Contents"
toc_icon: "cog"
header:
  overlay_image: /assets/images/tips.png
  overlay_filter: 0.5
  caption: python tips
---

Use decorator to log time when running an algorithm as follows. <br>

I usually records `ms` units  

```python
import time
def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        elapsed_time = (time.time() - start_time) * 1e3
        print("WorkingTime[{}]: {:.5f} ms".format(original_fn.__name__, elapsed_time))
        return result
    return wrapper_fn
```

Therefore, we can use this decorator as follows.

```python
@logging_time
def algorithm():
	...
```