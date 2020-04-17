---
title: "Window Anaconda setup"
excerpt: "describe how to use anaconda on window"

categories:
  - tips
tags:
  - install
  - WSL
use_math: true
last_modified_at: 2020-02-24
---

# Anaconda 설치법 
[Anaconda 설치설명 blog](http://taewan.kim/tutorial_manual/dl_pytorch/02/install/linux_env/)
blog를 보고 설치하면 된다.
요약 
```shell
#다운로드, offical 홈페이지에서 다운로드 받는것이 좋다.
$ curl -O https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh 
## 로그 생략 
$ ls -al
total 668868
drwxr-xr-x  6 ubuntu ubuntu      4096 Feb 19 01:10 .
drwxr-xr-x  4 root   root        4096 Feb 18 19:48 ..
-rw-rw-r--  1 ubuntu ubuntu 667822837 Feb 19 01:10 Anaconda3-5.3.0-Linux-x86_64.sh

$ bash Anaconda3-5.3.0-Linux-x86_64.sh

Please, press ENTER to ctntinue
>>>	Enter	설치 여부 확인
Do you accept the license terms? [yes|no][no]
>>>	yes	라이센스 동의
[/home/ubuntu/anaconda3] >>>	Enter	Anaconda 설치 위치 지정
Do you wish the installer to initialize Anaconda3
in your /home/ubuntu/.bashrc ? [yes|no]
[no] >>>	yes	bashrc 환변 변수 설정
Do you wish to proceed with the installation of Microsoft VSCode? [yes|no]>>>	no	vscode 설치

설치 활성화 
$ source ~/.bashrc
```



`$ conda --version`
path를 잡지 못한다면, path 설정을 `vim ~/.bashrc` 에 가서 해주고, `source ~/.bashrc` 를 통해 적용을 한다. 

~/.bashrc 파일 안에 `export PATH="/home/swyoo/anaconda3/bin:$PATH"`를 덭붙혀 준다.

`$ which python` 했을떄, 
`/home/swyoo/anaconda3/bin/python` 이렇게 나와야하며, 

conda list  했을때 설치 파일들이 나와야한다. 


### conda 가상환경 
`$ conda info --env` 명령시, conda environments를 몰수 있다. 

`$ conda create -n <가상환경 이름> python=<버전정보>`
e.g., `conda create -n swyoo python=3.6` 을 통해 python3.6 환경에서 작동하는 virtual environment를 구성할 수 있다. 

`$ conda env remove -n <가상환경 이름> `  

`$ conda remove -name <가상환경 이름> --all`

가상환경을 지울수 있다. 

`$conda activate <가상환경 이름>`
`$conda deactivate `

### 가상환경 kernel 적용
가상환경에서 ... [link](https://tech.songyunseop.com/post/2016/09/using-jupyter-inside-virtualenv/)
`$ pip install ipykernel`  
`$ python3 -m ipykernel install --user --name=<가상환경 이름>` # 명령어 한 줄로 kernel 등록
`jupyter notebook` 접속
Kernel> change kernel> <가상환경 이름>

### 버전에 맞는 modul 설치 
e.g., `$ pip install tensorflow-gpu==1.8.0`  
만약, downgrade 또는 upgrade를 원한다면,  
`$ pip install --upgrade tensorflow-gpu==1.4.0`  
이런식으로 사용하면 된다
