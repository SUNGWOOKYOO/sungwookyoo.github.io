---
title: "How to implement progressive bar on python fantastically"
excerpt: "simple tutorial of implement progressive bar on python"
categories:
 - tips
tags:
 - python
use_math: true
last_modified_at: "2020-05-08"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/teaser.jpg
 overlay_filter: 0.5
---

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
import time
```

</div>

# Print Progressive Bar in python

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
# Print iterations progress
def printProgressBar(iteration, total, msg, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    # print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    show = '\r{}|{}| {} {}% - {}'.format(prefix, bar, percent, suffix, msg)
    print(show, end='\r')
```

</div>

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
num_step = 5000
for i in range(num_step):
    time.sleep(0.01)
    printProgressBar(iteration=i + 1, total=num_step, msg='iteration', length=50)
```

</div>

{:.output_stream}

```
|██████████████████████████████████████████████████| 100.0 % - iteration
```

## logging time

Using decorator, we can implement logging time function. 

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        verbose = False
        if 'verbose' in kwargs:
            verbose = True
            del kwargs['verbose']
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        elapsed_time = (time.time() - start_time) * 1e3
        if verbose:
            print("WorkingTime[{}]: {:.5f} ms".format(original_fn.__name__, elapsed_time))
            return result
        return result, elapsed_time
    return wrapper_fn
```

</div>

<div class="prompt input_prompt">
In&nbsp;[10]:
</div>

<div class="input_area" markdown="1">

```python
@logging_time
def func():
    for i in range(num_step):
        time.sleep(0.001)
        printProgressBar(iteration=i + 1, total=num_step, msg='iteration', length=50)
```

</div>

<div class="prompt input_prompt">
In&nbsp;[11]:
</div>

<div class="input_area" markdown="1">

```python
func(verbose=True)
```

</div>

{:.output_stream}

```
WorkingTime[func]: 6094.05684 ms███████████████████| 100.0 % - iteration

```
