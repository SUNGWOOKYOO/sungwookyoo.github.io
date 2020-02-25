---
title: "Jupyter setup"
excerpt: "setup tips for jupyter"

categories:
  - tips
tags:
  - jupyter
  - install
use_math: true
last_modified_at: 2020-02-24
toc: true
toc_label: "Contents"
toc_icon: "cog"
toc_sticky: true
---

# 소스 코드 작성 요령

버전에 맞는 library를 사용해야 dependency 문제가 없다.  따라서, 다음과 같은 버전체크와 path를 명시

```python
'''
Created on July 1, 2019
@author: SungWook Yoo 
'''
import tensorflow as tf
from platform import python_version
import os
!type python
print(python_version())
print(tf.__version__)
# path = os.getcwd() + '/Data/ml-1m'
# os.path.join(<base directory>, <추가할 디렉토리 이름>)
path = os.path.join(os.getcwd(), 'Data/ml-1m')
path
```



# jupyter notebook 원격접속 서버 설정 

### prerequsite: juptyer가 설치 되어 있어야한다.

1. config 파일 만들기 

   `$jupyter notebook --generate-config`

   실행결과: `Writing default to : ...# <= 이 디렉토리에 jupyter_notebook_config.py 생성`

   `...` = `/home/swyoo/.jupyter/jupyter_notebook_config.py`

   rf. 연구실 컴퓨터의 경우 `/home/kddlab/.jupyter/jupyter_notebook_config.py`

2. ipython으로 비밀번호 설정

   ```python
   $ ipython #실행 후
   In [1]: from notebook.auth import passwd
   In [2]: passwd() 
   #입력하면 pw 입력하라고 뜬다, 그럼 입력해주세요
   Out [2]: '.......' #'...'의 ...은 본인의 패스워드 입니다. <= 복사해주세요!
   #ipython 종료
   exit()
   ```

3. 패스워드를 config하기 

   ```shell
   $ gedit /home/username/.jupyter/jupyter_notebook_config.py
   
   # 따옴표 ' '를 꼭 붙이기!  
   c.NotebookApp.ip = '내 ip'
   c.NotebookApp.password=u'sha1:....' 
   c.NotebookApp.open_browser = False #원래 True
   
   # 만약 원격 연결이 안된다면 외부 허용, 포트번호 바꿔보기
   c.NotebookApp.allow_origin = '*'
   c.NotebookApp.port = '사용할 포트번호 네자리를 입력해주세요, 초기값은 8888 입니다.'
   ```

4. `$ jupyter notebook ` 실행

###  jupyter notebook 또는 lab을 background[^1 ]로 돌리기

#### prerequsite: port가 설정되어있어야 `&` 를 통해 자동으로 port할당할 수 있다.  

```shell
# jupyter notebook을 background로...
$ jupyter notebook& 
# jupyter lab을 background로...
$ jupyter lab&

# 만약, background로 실행하면 종료가 되지 않기 때문에 따로 PID를 찾아 종료해 주어야한다.
# 현재 실행중인 jupyter notebook들 보기
$ jupyter notebook list
# 현재 실행중인 PID list
$ netstat -tulpn
# PID/Program name에서 <PID>에 해당하는 부분을 다음 명령어를 통해 kill 
$ kill <PID> 
```



[^1]: background 로 돌린다는 것은 평소대로라면 terminal이 종료될때 jupyter notebook은 종료되게 된다. terminal의 종료 유무와 상관없이 jupyter notebook 또는 lab이 동작하도록 하는것을 background로 실행하는것이다.  단, 이렇게 실행하면 jupyter notebook을 종료하는게 까다롭게 된다. 

 

###  anaconda로 설치한 가상 환경 jupyter notebook kernel에 추가하기
[link](https://data-newbie.tistory.com/113)

```shell
# kernel을 만들기위한 module
$ pip install ipykernel

# conda 의 가상환경이름이 virtualEnv이며, kernel 선택화면에서 보일 이름이 [displayKernelName]
python -m ipykernel install --user --name [virtualEnv] --display-name [displayKernelName]
# e.g
# python -m ipykernel install --user --name env --display-name swyoo

# 주피터 커널 리스트 보기
$ jupyter kernelspec list

# 주피터에 뜨는 커널 지우기
$ jupyter kernelspec uninstall yourKernel
```



<font color=red> Note that </font> `!type python`을 해도 주피터 노트북 상에서는 base에 대한 python 을 보여준다. 하지만 제대로 동작함을 확인할 수 있다.



### 가상환경 추출
: 모든 세팅이 되어 있는 가상환경을 다른 머신으로 복사하고 싶을 때 사용하면 된다. 아래 명령어는 현재 환경을 *environment.yml* 파일로 저장한다.

```shell
$ conda env export --name YOUR_ENV_NAME > environment.yml
```

### 추출한 가상환경으로 새로운 가상환경 생성
: 앞서 추출한 *environment.yml* 로 가상환경을 생성한다. 설치되어 있던 모든 패키지가 자동으로 설치된다.

```shell
$ conda env create -f ./environment.ym
```



### Tip 

vi editor에서 원하는 text 찾을때, 명령어 모드에서  

` :?<검색어> `# 윗방향으로 검색 

`:/<검색어>` # 아랫방향 검색

`n` 누르면 다음 단어

`u` 누르면 이전 되돌리기

`ctrl + F6`  강제종료  나중에 *.swp 폴더도 지워주어야함









