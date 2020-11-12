---
title: "파일 관리와 관련된 리눅스 명령어"
excerpt: "learn about linux command for file management"
categories:
 - tips
tags:
 - linux
use_math: true
last_modified_at: "2020-11-12"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/teaser.jpg
 overlay_filter: 0.5
---

# 파일 관리 명령어



이번 포스팅에서는 간단하게 파일 관리에 관한 명령어들을 정리해 보았따. 



## Linux 디스크 용량 확인(df/du)

```shell
# 현재 머신에서 PATH에 있는 디스크들의 남은 용량을 보기좋게 보여줌 
$ df -h 
# 현재 디렉토리에서 서브디렉토리까지의 사용량 확인 
$ du -h 
```



## 폴더 내부 구조를 확인(tree)

```shell
# tree 라는 명령어를 이용하여 파일 구조를 확인, 다음 줄은 사용 예시
$ tree -h Algorithm/
Algorithm/
├── [4.0K]  bin
│   └── [ 529]  Solution.class
└── [4.0K]  src
    └── [ 150]  Solution.java
```



## 머신의 스토리지 파티션 확인

```shell
# 현재 머신에 달려있는 스토리지들 확인
$ fdisk -l
# 결과로 /dev/[스토리지 이름]: [스토리지 사이즈]Gib 형식과 밑에 파티션 정보들이 출력된다.
```

