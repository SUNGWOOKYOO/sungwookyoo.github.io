---
title: "Unable to display frame varibles in Pycharm"
excerpt: "Collecting data... in pycharm "
categories:
 - tips
tags:
 - pycharm
use_math: true
last_modified_at: "2020-03-05"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
  overlay_image: /assets/images/teaser.jpg
  overlay_filter: 0.5
  caption: I want to go to space someday.
  actions:
   - label: "stack overflow"
     url: "https://stackoverflow.com/questions/40865488/why-does-pycharm-say-unable-to-display-frame-variables-in-debug-mode"

---

# How to solve it?
Pycharm 에서 종종 debugging을 할때, `Collecting data...` 가 지속되다가 
`Unable to display frame variables` 가 뜨는 경우가 있다. 

이경우 간단히 해결할 수 있다.

`File > settings > Build, Execution, Deployment > Python Debugger >
Gevent compatible`
을 체크하면 디버거가 다시 잘 동작한다. 

