---
title: "GPU setup"
excerpt: "setting for GPU"

categories:
  - tips
tags:
  - install
  - GPU
use_math: true
last_modified_at: 2020-02-24
---



GPU setting 방법 

[연구실 guide](http://kdd.snu.ac.kr/wiki/index.php/GUIDE:GPU)



### 상식

GPU를 돌리기에 앞서 나의 gpu 머신이 어떤 graphic driver와 호환되는지 알아야한다. 

graphic driver는 cuda기반 programming으로 돌아가기 때문에, graph driver와 호환이 맞는 cuda, cudnn 등을 깔아야하며, tensorflow-gpu역시 호환이 맞아야한다.

다음과 같이 구글에 검색하여 참조한다. 

```shell
# google search keyword 
tensorflow source build
```



# Install CUDA 

다음과 같이 다운로드 받고싶은 cuda version을 검색한다. [검색 결과](https://developer.nvidia.com/cuda-10.0-download-archive)

```shell
# google search keyword
cuda toolkit 10.0
```
다음과 같이  머신의 운영체제의 환경에 맞는 선택을 한다.
<img src='/assets/images/cuda_install.PNG'/>

`deb(local)` 을 선택하고,  installation Instructions를 따라 설치한다. 

```shell
# 설치 예시 
# prerequsite: download 받은 폴더로 가야한다.
$ sudo dpkg -i cuda-repo-ubuntu1604-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
# Instruction에서 <version>을 대체한 부분으로 /var/[tab] 을 통해 확인후 복사 붙여넣기.
$ sudo apt-key add /var/cuda-repo-10-0-local-10.0.130-410.48/7fa2af80.pub
# OK 뜬다면 성공
$ sudo apt-get update
# 지정해서 설치하고싶을땐 cuda=<version> 여기서 <version>은  Tab 누르면 알수있다,
# e.g, sudo apt-get install cuda=10.1.105-1 
$ sudo apt-get install cuda
# 설치후 확인
$ nvcc --version 
```

<font color=red> Warning: </font> Other installation options are available in the form of meta-packages. For example, to install all the library packages, replace "cuda" with the "cuda-libraries-10-1" meta package. For more information on all the available meta packages click [here](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-metas).



`nvcc --version` 이 잘 동작하지 않을 경우 

`~/.bashrc` 를 편집하여 path를 잡게 하면된다.

```shell
#################################################################
# ~/.bashrc 내부 ...
#################################################################
export PATH=/usr/local/cuda-<version>/bin${PATH:+:${PATH}}

# example
export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
#################################################################
# 파일에서 나와서
$ source ~/.bashrc
```



CUDA를 설치하면서 CUDA 버전에 맞는 NVIDIA Driver가 기존 드라이버에서 업그레이드 될 수 있는데,  

업데이트된 Driver를 다음과 같이 확인 할 수 있다.

```shell
# 만약 잘 작동하지 않는다면, reboot를 먼저 해보자!
# (384 -> 410 으로 업데이트 될때 reboot를 해야 바뀌었었다.)
$ nivida-smi
```



## Install multiple CUDA version

cuda version을 여러개 설치해 놓고,  필요할 떄마다 바꾸고 싶은 경우, 다음과 같은 방식으로 설치한다.

<font color=red>`Warnning:` </font> NVIDIA Driver는 CUDA를 설치하면서 업데이트 될 수 있으므로 조심. 

일단 기존에 설치되어 있는 CUDA는 내버려둔 상태에서 설치를 진행한다.

```shell
# 받고싶은 CUDA toolkit version 검색하여 설치
# google search keyword
cuda toolkit 9.0
```

앞전에 설치했던 방식과 동일하게 머신의 환경에 맞는 CUDA를 설치한다.

<font color=red> 중요:</font> 설치 후,  환경 PATH를 잘 설정 해 주어야 원하는 CUDA를 선택적으로 사용 할 수 있다.

```shell
# 환경 변수를 편집하기 위해 bashrc 폴더에 접근
$ gedit ~/.bashrc
...
######################################################################################
# ./bashrc 파일 내부
######################################################################################
# alias를 설정하므로서, 원하는 cuda를 설정할 수 있게 된다.
alias cuda9='export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}} \
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}'
alias cuda10='export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}} \
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}'
######################################################################################
# 저장 후, gedit을 종료후에 실행
$ source ~/.bashrc
```

`alias` 명령어를 통해 alias가 잘 설정되어있나 확인을 한다.

`nvcc --version` 명령어를 통해 현재 path로 잡혀있는 cuda의 version을 확인한다. 

```shell
# cuda 9.0을 쓰고 싶은 경우, 
$ cuda9
# cuda 10.0을 쓰고 싶은 경우, 
$ cuda10
```





# Install NVIDA Driver

보통은 CUDA를 설치하면 자동으로 설치된다.

[NVIDA Driver](https://www.nvidia.com/Download/index.aspx) 에서 직접 다운로드 받아 설치할 수 있다.

[한글 블로그 설명](https://codechacha.com/ko/install-nvidia-driver-ubuntu/) 에서 자세한 방법을 볼 수 있다.

크게 2가지 방법존재. 

### 

```shell
# Display Manager stop
$ systemctl isolate multi-user.target
$ sudo service lightdm stop 

# Display Manager start
$ systemctl start multi-user.target
$ sudo service lightdm start 
```



### Install CUDNN

[version download link](https://developer.nvidia.com/rdp/cudnn-archive) 에 들어가면 버전별로 설치 파일들이 나열되어 있다.

설치하는 방법은 다음과 같이 3가지가 있다.

1. `*.tgz` file을 이용한 방법 [cuDNN Library for Linux]
2. `*.deb` 파일을 이용한 방법 [Deb]
3.  이 외의 방법[RPM]

수동으로 설치하는 것이 더 쉬울 수 있어 `*.tgz` 파일을 압축 해제한뒤 cuda 의 path부분에 복사해 넣는 방식인 첫번째 방식을 택해 설치 하였다.

[official guide](https://docs.nvidia.com/deeplearning/sdk/pdf/cuDNN-Installation-Guide.pdf) 에서 **2.3.1. Installing From A Tar File** 부분을 보고 설치하였다.

먼저 `*.tgz` 파일을 현재 디렉토리 (보통은` ~/Download`)에 다운로드 받고 압축을 해제한다.

```shell
# prerequsite: download cudnn-10.0-*.tgz 
$ cd ~/Download
# unpack this file.
$ tar -xzvf cudnn-10.0-*.tgz
# then, being generated './cuda' folder

# prerequite: should be exist '/usr/local/cuda<version>' path 
# copy some files to [cuda path]/[proper places]
$ sudo cp cuda/include/cudnn.h /usr/local/cuda<version>/include
$ sudo cp cuda/lib64/libcudnn* /usr/local/cuda<version>/lib64
$ sudo chmod a+r /usr/local/cuda<version>/include/cudnn.h /usr/local/cuda<version>/lib64/
libcudnn*

# e.g 
$ sudo cp cuda/include/cudnn.h /usr/local/cuda-10.0/include
$ sudo cp cuda/lib64/libcudnn* /usr/local/cuda-10.0/lib64
$ sudo chmod a+r /usr/local/cuda-10.0/include/cudnn.h /usr/local/cuda-10.0/lib64/
libcudnn*

# reboot 
$ sudo reboot 
```



```shell
# version check
$ cat /usr/local/cuda<version>/include/cudnn.h | grep CUDNN_MAJOR -A 2
# e.g 
$ cat /usr/local/cuda-10.0/include/cudnn.h | grep CUDNN_MAJOR -A 2
```

 

## tensorflow error

cudnn version 호환이 tensorflow-gpu 와 맞지 않은경우, 중간에 kernel이 die되거나

cudnn 관련 error발생한다.





### Path 잡기

```shell
# vi ~/.bashrc
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```



# Uninstall CUDA & CuDNN

### PRE-INSTALLATION ACTIONS

기존에 설치되어 있는 것들을 제거하는 과정. 설치되어 있는가를 확인하려면

```shell
$ apt list --installed | grep nvidia
$ apt list --installed | grep cuda
```

제거하는 명령어:

```shell
$ sudo apt-get --purge remove '^cuda.*'
$ sudo apt-get --purge remove '^nvidia.*'
$ sudo apt-get --purge remove '^libcudnn7.*'
$ sudo apt-get --purge remove '^libnvidia-.*'
```

apt로 설치되지 않는 것들	은 다음과 같이 제거해야 한다.

```shell
$ sudo /usr/local/cuda/bin/uninstall_cuda_X.Y.pl
$ sudo /usr/bin/nvidia-uninstall
```

마지막으로 제대로 제거되었는지 확인한다

```shell
$ ls -d /usr/local/cuda*
```

제거되지 않았으면 제거한다.