---
title: "Linux tips"
excerpt: "Linux 를 다루는데 필요한 tip"

categories:
  - tips
tags:
  - linux
use_math: true
last_modified_at: 2020-02-24
---


## 우분투 version check 

```shell
$ lsb_release -a
```



## 환경변수 추가 후 적용하는 코드 

환경변수를 추가하는 file은 보통 `~/.bashrc` 파일이다. 

따라서, `gedit ~/.bashrc` 를 실행 후, 아나콘다 가상환경을 activate 하기 위한 path를 적어논다고하면, 

`export PATH="/home/kddlab/anaconda3/etc/profile.d/conda.sh"` 와 같이 적을수 있고, 

적은 후에 shell script에서 
```shell
# 이렇게 command line을 치면 적용이 완료됨.
$ source ~/.bashrc 
```



## 파일에 대한 alias 를 설정하기

`~/.bashrc`파일에 alias를 설정해두면 파일에 직접가지 않더라도 바로 실행할 수 있게 된다. 

따라서, `gedit ~/.bashrc` 를 실행 후, pycharm을 실행하기 위한 alias를 만든다고하면,

다음의 예시와 같은 line을 추가 할 수 있다.

`alias pycharm="/home/kddlab/Downloads/pycharm-community-2019.2.2/bin/pycharm.sh"` 

```shell
# 바뀐 변수를 적용을 시킨뒤
$ source ~/.bashrc
# 실행하면 sh ~/pycharm.sh 와 같은 효과를 볼 수 있다.
$ pycharm 
```



## 바로가기 만들기
```shell
ln -s [path]
```



## sudo apt-get update 에러시

다음과 같은 명령어를 한 후, 다시 시도해본다. [stack overflow](https://askubuntu.com/questions/760574/sudo-apt-get-update-failes-due-to-hash-sum-mismatch)

```shell
$ sudo apt-get clean
$ sudo rm -r /var/lib/apt/lists/*
```

 update, upgrade, autoremove 를 번갈아 가면서 하다보면 해결되는 경우가 많다.

```shell
$ sudo apt-get update
```



### 또 다른 에러상황 

```shell
E: Failed to fetch http://ppa.launchpad.net/texlive-backports/ppa/ubuntu/dists/xenial/main/binary-amd64/Packages  404  Not Found [IP: 91.189.95.83 80] 
와 같은 에러를 보았다면 
# 다음과 같이 해보도록 한다.
$ sudo add-apt-repository --remove ppa:texlive-backports/ppa 
$ sudo apt update
```



### 또 다른 에러상황[GPG 에러]

[StackOverflow](https://askubuntu.com/questions/943146/apt-update-error-an-error-occurred-during-the-signature-verification-chrome)

```shell
W: An error occurred during the signature verification. The repository is not updated and the previous index files will be used. GPG error: http://dl.google.com stable Release: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY 78BD65473CB3BD13

$ wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add -

# OK 가 뜬다면 성공
```



## Linux 디스크 용량 확인(df/du)

```shell
# 디스크의 남은 용량을 보기좋게 보여줌
$ df -h
# 현재 디렉토리에서 서브디렉토리까지의 사용량 확인
$ du -h
# 또 다른 옵션
$ sudo du -sh *
```



## 파일 다운로드 받기

`$ curl -o [파일이름] [url]`  이때,  url은 raw data이어야한다

```shell
# example
$ curl -o train-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
```



## 외장 하드 마운트 시키기 

[설명 블로그](https://m.blog.naver.com/kimmingul/220639741333)

```shell
# 현재 장착된 하드디스크 목록을 확인할 수 있다.
$ sudo fdisk -l
# 여기서 /dev/sda, /dev/sdb, /dev/sdc...  이렇게 기술된 부분이 물리적인 하드디스크를 말하며, /dev/sda1 ... 등 1,2,3.. 숫자가 붙으면 각 하드디스크별 파티션이라고 보면 된다.

$ mkdir /data
$ chmod 777 /data  # 권한 변경
$ mount /dev/sdb1 /data
$ mount 

# 확인
$df -h 

# 영구 마운트 시키기
# blkid 명령어를 이용하여 UUID 값을 확인해볼 수 있다.
$ sudo blkid

# 이제 /etc/fstab를 편집한다.
$ gedit /etc/fstab
UUID=[UUID] [마운트할 디렉트리] [타입]    defaults        1 2
# example
# UUID=f30bcefe-e166-4526-ad1c-6a84e85dca69 /data ext4    defaults        1 2
$ sudo mount -a 
$ df -h

# 재부팅후 다시 마운트 확인
$ sudo reboot 
```