---
title: "How to use GUI on Docker"
excerpt: "Let's learn about how to use GUI on docker"
categories:
 - tips
tags:
 - docker
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

#  Docker GUI



## 원격 서버에서 docker 사용시 GUI사용
ssh 를 통해 원격 서버에 접속하면 DISPLAY 변수가 `:10.0`(`[host]:<display>[.screen]`의미)이 아니라 다음과 같이 된다. 

```shell
# example
$ echo $DISPLAY
localhost:10.0
```



얼핏 생각하면 원격 서버에서 docker를 실행할때 다음과 같이 DISPLAY 환경변수를  현재 접속한 원격 정보로 주어 x11 forwarding을 통해 해결할 수 있을 것 같았다. 

```shell
$ docker run -it -e DISPLAY=$DISPLAY name_of_docker_image
```

하지만 GUI 를 사용하는 애플리케이션을 실행해본 결과 다음과 같이 에러 발생.

```
(gedit:17): Gtk-WARNING **: 09:17:53.515: cannot open display: localhost:11.0
```







docker에서 X11 forwarding 을 통해 gui를 사용해야한다. 

/etc/ssh/sshd_config 다음 옵션 수정 [참고자료: https://unix.stackexchange.com/questions/403424/x11-forwarding-from-a-docker-container-in-remote-server]

> 1. X11Forwarding
>
>    X11 전송을 허가하는지 어떤지를 지정합니다. 디폴트는 "no" 입니다. X11 전송을 금지해도 보안를 올리는 것은 전혀 없는 것에 주의해 주세요. 왜냐하면 유저는 언제라도 자기 부담의 전송 프로그램을 인스톨 해 사용할 수가 있기 때문입니다. UseLogin하지만 허가되고 있으면(자), X11 전송은 자동적으로 금지됩니다.
>
>    출처: https://thinkfarm.tistory.com/entry/sshdconfig-설명 [thinkfarm]
>
> 2. X11UseLocalhost
>
>    sshd 하지만 전송 된 X11 서버를 루프백 주소 (localhost)에 bind 하는지 어떤지를 지정합니다. 디폴트에서는, sshd (은)는 전송 된 X11 를 루프백 주소에 bind 해, 환경 변수 DISPLAY 의 호스트명의 부분을 "localhost" (으)로 설정합니다. 이렇게 하면(자), (역주: SSH 서버 이외의) 리모트 호스트로부터 전송 된 X서버에 접속할 수 없게 됩니다. 그러나, 낡은 X11 클라이언트라고, 이 설정에서는 동작하지 않는 것이 있습니다. 그러한 때는X11UseLocalhost (을)를 "no" (으)로 설정해, 전송 된 X 서버가 와일드 카드 주소에 bind 되도록 할 수 있습니다. 이 옵션의 인수는 "yes" 혹은 "no" 입니다. 디폴트에서는, 이것은 "yes (localhost 밖에 bind 하지 않는다)" (이)가 되어 있습니다.
>
>    출처: https://thinkfarm.tistory.com/entry/sshdconfig-설명 [thinkfarm]

```
X11Forwarding yes
X11UseLocalhost no
```



[참고자료: https://stackoverflow.com/questions/48235040/run-x-application-in-a-docker-container-reliably-on-a-server-connected-via-ssh-w]

```
XAUTH=~/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
X11PORT=`echo $DISPLAY | sed 's/^[^:]*:\([^\.]\+\).*/\1/'`
TCPPORT=`expr 6000 + $X11PORT`
sudo ufw allow from 172.17.0.0/16 to any port $TCPPORT proto tcp
DISPLAY=`echo $DISPLAY | sed 's/^[^:]*\(.*\)/172.17.0.1\1/'`
```



도커 실행할때 환경변수 설정과 xauth 파일 볼륨으로 전달

```
docker run \
  -e DISPLAY=${DISPLAY} \
  -e XAUTHORITY=$XAUTH \
  -v $XAUTH:$XAUTH:ro \
```





docker run -it -e DISPLAY=$TEMP_DISPLAY -e XAUTHORITY=$XAUTH -v $XAUTH:$XAUTH:ro 7951490944d2
