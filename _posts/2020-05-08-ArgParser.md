---
title: "How to use argparse and yaml module on jupyter at the same time"
excerpt: "simple tutorial of argparse and yaml"
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
In&nbsp;[1]:
</div>

<div class="input_area" markdown="1">

```python
import sys, os, argparse, yaml
```

</div>

# How to use argparse and yaml

## Basic usage of argparse in jupyter notebook
When using argparse module in jupyter notebook, all `required` flag should be `False`. <br>
Before calling `parser.parse_args()`, we should declare as follows.
```python
sys.argv = ['-f']
```

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
parser = argparse.ArgumentParser()
parser.add_argument('-config', help="configuration file *.yml", type=str, required=False, default='config.yml')
parser.add_argument('-args', help="learning rate", type=bool, required=False, default=False)
# training parameters
parser.add_argument('-epochs', help="num of epochs for train", type=int, required=False, default=100)
parser.add_argument('-lr', help="learning rate", type=float, required=False, default=0.00005)
parser.add_argument('-batch_size', help="batch size", type=int, required=False, default=64)
sys.argv = ['-f']
args = parser.parse_args()
```

</div>

## Use yaml and argparse at the same time

>If you want to know about `Loader` option in pyyaml, see [this document in StackOverflow](https://stackoverflow.com/questions/55677397/why-does-pyyaml-5-1-raise-yamlloadwarning-when-the-default-loader-has-been-made)

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
%%writefile config.yml
epochs: 100
lr: 0.001
batch_size: 50
```

</div>

{:.output_stream}

```
Overwriting config.yml

```

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
parser = argparse.ArgumentParser()
parser.add_argument('-config', help="configuration file *.yml", type=str, required=False, default='config.yml')
parser.add_argument('-args', help="learning rate", type=bool, required=False, default=False)
# training parameters
parser.add_argument('-epochs', help="num of epochs for train", type=int, required=False, default=100)
parser.add_argument('-lr', help="learning rate", type=float, required=False, default=0.00005)
parser.add_argument('-batch_size', help="batch size", type=int, required=False, default=64)
sys.argv = ['-f']
args = parser.parse_args()

if not args.args:  # args priority is higher than yaml
    opt = vars(args)
    args = yaml.load(open(args.config), Loader=yaml.FullLoader)
    opt.update(args)
    args = opt
else:  # yaml priority is higher than args
    opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
    opt.update(vars(args))
    args = opt
print("arguments: {}".format(str(args)))
```

</div>

{:.output_stream}

```
arguments: {'config': 'config.yml', 'args': False, 'epochs': 100, 'lr': 0.001, 'batch_size': 50}

```
