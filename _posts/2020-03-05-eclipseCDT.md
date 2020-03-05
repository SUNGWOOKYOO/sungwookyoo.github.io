---
title: "How to install Eclipse CDT on Ubuntu 16.04"
excerpt: "우분투에서 Eclipse CDT 설치 방법 가이드"
categories:
 - tips
tags:
 - eclipse
use_math: true
last_modified_at: 2020-03-05
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/eclipse.png
 overlay_filter: 0.5
 caption: how to install Eclipse CDT on Ubuntu
 actions:
  - label: "Eclipse"
  - url: "https://www.eclipse.org/cdt/"
---

# Eclipse CDT install

[blog guide](https://agiantmind.tistory.com/182)

tar file을 통해 설치하는것이 편하다.

1. JDK 설치 

   ```shell
   # version 확인
   $ java --version
   ```

2. C/C++ 개발관련 package 설치

   ```shell
   $ sudo apt-get install build-essential
   ```

3. [offical site](http://www.eclipse.org/) 에서 `Eclipse IDE for C/C++ Developers` 다운로드

   ```shell
   # check os architecture 
   # 64비트 - amd64, x86_64
   $ uname -m
   ```

4. guide

   ```shell
   # STEP1. 다운받은 경로 이동 후 압축 해제
   $ tar xvzf [eclipse-file.tar]
   
   # STEP2. eclipse 폴더를 /opt 로 이동
   $ sudo mv eclipse /opt
   # 이제 ./opt/eclipse/eclipse를 통해 실행 가능해진다.
   
   # STEP3. 터미널에서 실행 될 수 있도록 설정
   $ sudo vi /usr/bin/eclipse
   #############################
   #! /bin/sh
   export ECLIPSE_HOME=/opt/eclipse
   $ECLIPSE_HOME/eclipse $*
   #############################
   # STEP4. 바로가기 설정에 대한 권한 설정
   $ sudo chmod 755 /usr/bin/eclipse
   
   # STEP5. Desktop에서 바로가기 파일 생성
   $ sudo vi /usr/share/applications/eclipse.desktop 
   ##############################
   [Desktop Entry]
   Encoding=UTF-8
   Name=Eclipse
   Comment=Eclipse IDE
   Exec=/opt/eclipse/eclipse # eclipse 실행 파일 지정!
   Icon=/opt/eclipse/icon.xpm
   Terminal=false
   Type=Application
   Categories=Development
   StartupNotif=true
   ##############################
   ```

   



# C+11 문법 사용하기

* C/C++ Build -> Settings -> Tool Settings -> GCC C++ Compiler -> Miscellaneous -> Other Flags. Put `-std=c++0x` (or for newer compiler version `-std=c++11` and apply and close.

* C/C++ General > Preprocessor Include Paths, Macros etc. 

  Providers > check `CDT GCC Build-in Compiler Settings`.

  그리고 하단에 "Use global provider shared between projects" 선택을 해제하고,

  Command to get compiler specs 칸에 **-std=c++0x** 를 추가한다.

  또한 순서상 중간 쯤에 와 있는 CDT GCC Build-in Compiler Settings 항목을 Move Up을 이용해서 가장 위로 끌어올린 뒤에 Apply 버튼과 OK 버튼을 누른다.

  <img src="https://t1.daumcdn.net/cfile/tistory/2545C633581083A528" width=600>

  [blog](https://skylit.tistory.com/247)

  

#  
