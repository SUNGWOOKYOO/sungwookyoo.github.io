---
title: "Git Collboartion"
excerpt: "How to collaborate with others on git"
categories:
 - tips
tags:
 - git
use_math: true
last_modified_at: "2020-11-11"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/teaser.jpg
 overlay_filter: 0.5
---

# Git Collaboration

![github2.png](https://image.toast.com/aaaadh/real/2017/techblog/github2.png)

GitHub를 통한  collaboration은 다음과 같은 다이어그램으로 이루어진다. 

개발과정에 대한 flow 는 다음과 같이 진행된다.

보통  upstream이 한 프로젝트의 기둥이 되고, 프로그래머는 자기가 맡은 모듈을 upstream에서 fork 해와서 origin으로 둔다.

위의 upstrream, origin은 git의 원격 저장소이다. 

이제, origin에서 local로 clone해와서 개발을 진행하게 된다. 

프로그래머가 맡은 모듈에 개발이 완료가 되면 local에서 origin으로 push하고, origin 을 Pull Request 하므로서, 코드리뷰가 진행된다.

리뷰가 완료된 후에 upstream에 origin을 프로젝트의 리더가 merge 한다. 



## Fork한 origin을 upstream의 최신 버전으로 동기화

개발 프로젝트가 진행중인 상황에서  업데이트된 upstream를 최신버전으로 동기화 하고 싶을 떄가 많다. 

> Prerequisite: local에 remote 에대한 주소가 등록되어있어야한다. 
>
> 예시로 `git remote -v` 명령을 사용하면 원격 리포지토리에 대한 등록 정보를 확인할 수있다.
>
> [root@6e031d1a6b39 log_partitioner (log-partitioner)]# git remote -v
> origin  https://github.daumkakao.com/tony-yoo/contextual_bandit_pilot.git (fetch)
> origin  https://github.daumkakao.com/tony-yoo/contextual_bandit_pilot.git (push)
> upstream        https://github.daumkakao.com/toros/contextual_bandit_pilot.git (fetch)
> upstream        https://github.daumkakao.com/toros/contextual_bandit_pilot.git (push)

이때 만약 upstream에 대한 정보가 없거나 추가적으로 친구나 동료의 fork 된 프로젝트를 원격 저장소로 등록하고 싶은 경우에는 아래와 같이 추가할 수 있다.

```bash
$ git remote add upstream https://github.com/projectusername/repo.git
# add dave repo.
$ git remote add dave https://github.com/dave/repo.git
```



다음의 명령어를 통해 upstream의 최신 정보들을 업데이트 한다. 

```shell
$ git fetch upstream
```

upstream이 최신 버전으로 업데이트 되었으니, clone 된 local 을 upstream과 merge 하고 fork해둔 origin을 업데이트한다.

```shell
# local/master branch is updated.
$ git checkout master    
$ git merge upstarem/master
# origin/mastesr branch is updated by local/master.
$ git push origin master
```





