---
title: "byobu setup"
excerpt: "tips for byobu"

categories:
  - tips
tags:
  - byobu
use_math: true
last_modified_at: 2020-02-24
---

### window 에서 unbuntu 사용하기 및 설치
[설치 및 삭제 방법](https://www.howtoinstall.co/en/ubuntu/xenial/byobu?action=remove)

### byobu 사용
[byobu 사용시 발생 문제 stack overflow](https://askubuntu.com/questions/492802/byobu-weird-character)

```shell
$ byobu-config
```
에서 "Toggle Status Notifications", de-selecting "logo", and then pressing "Apply".

### byobu session 다루기
```shell
# 누가 session을 만들었는지 본다.
$ byobu list-session
# 내 세션을 만든다. 
$ byobu new -s <session_name> 
e.g. $ byobu new -s swyoo
```


### 바로가기 만들기 
ln -s <경로>
e.g., ln -s /mnt/c/Git

## Tip
ctrl + s 누르면 화면이 멈추는데 ctrl + q 를 누르면 풀림  
ctrl + d 누르면 현재 보고있는 byobu 화면 하나를 닫는다.
F6 누르면 현재 session을 background로 둔채 byobu 나감  
F7 누르면 위로 커서 올려서 볼 수 있음  
ctrl + F6: session 강제종료    
shift + F2: 화면 수평으로 이등분
ctrl + F2: 화면 수직으로 이등분

## Link
[정리가 잘 되어있는 블로그](https://eungbean.github.io/2018/08/29/gpu-monitor-with-byobu/)
[session 다루기 stackoverflow](https://askubuntu.com/questions/196290/name-a-byobu-session)