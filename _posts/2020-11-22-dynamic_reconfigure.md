---
title: "Dynamic reconfiguration in ROS"
excerpt: "ros 에서 rqt를 이용하여 interactive하게 paramter를 setting할 수 있는 방법을 알아보자"
categories:
 - tips
tags:
 - ros
use_math: true
last_modified_at: "2020-11-22"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/teaser.jpg
 overlay_filter: 0.5
 caption: #
 actions:
  - label: "#"
    url: "#"
---
# Dynamic reconfiguration in ROS



## 1. Create package

```
cd ~/catkin_ws/src
catkin_create_pkg rospy
```



## 2. cfg 

```
cd ~/catkin_ws/src/dynamic_reconfig_test
mkdir config
cd config
gedit DynamicParam.cfg
```

write down as follows:

```
#!/usr/bin/env python
PACKAGE = "dynamic_reconfig_test"

from dynamic_reconfigure.parameter_generator_catkin import *

gen = ParameterGenerator()

# 'name', 'type', 'level', 'description', 'default', 'min', 'max
gen.add("test_parameter", int_t, 0, "A test parameter", 0, 0, 100)

# PACKAGE는 위에서 명시해 주었고, 두 번째 인자는 node명, 세 번째 인자는 생성되는 파일 앞에 붙는 접두어
exit(gen.generate(PACKAGE, "dynamic_reconfig_test", "DynamicParam"))
```

allow to write

```
chmod a+x config/DynamicParam.cfg
```



## 3. setup.py

```
cd ~/catkin_ws/src/dynamic_reconfig_test
gedit setup.py
```

write down as follows:

```
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
   packages=['dynamic_reconfig_test'],
   package_dir={'': 'src'})

setup(**d)
```



## 4. CMakeLists.txt

```
cmake_minimum_required(VERSION 3.0.2)
project(dynamic_reconfig_test)

find_package(catkin REQUIRED COMPONENTS
  rospy
  dynamic_reconfigure
)

catkin_python_setup()

generate_dynamic_reconfigure_options(
  config/DynamicParam.cfg
)

catkin_package(
# INCLUDE_DIRS include
# LIBRARIES dynamic_reconfig_test
# CATKIN_DEPENDS rospy
# DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(
  PROGRAMS src/dynamic_reconfig_test/dynamic_reconfig_test.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )

install(DIRECTORY config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
```



## 5. Node 

```
cd ~/catkin_ws/src/dynamic_reconfig_test/src/dynamic_reconfig_test
gedit dynamic_reconfig_test
```

write down as follows:

```
#!/usr/bin/python

import rospy
from dynamic_reconfigure.server import Server
from dynamic_reconfig_test.cfg import DynamicParamConfig

def callback(config, level):
    rospy.loginfo("Dynamic parameter: {}".format(config.test_parameter))
    return config

def main():
    rospy.init_node("dynamic_reconfig_test", anonymous=True)

    reconfigure_server = Server(DynamicParamConfig, callback=callback)

    rospy.loginfo("initialize the dynamic_reconfig_test node...")

    rospy.spin()

if __name__ == "__main__":
    main()
```

build

```
cd ~/catkin_ws
carkin_make
```



import error handling

import dynamic_reconfig_test 를 했을때,  라이브러리 경로 (~catkin_ws/devel/lib/python2.7/dist-packages/dynamic_reconfig_test/__init__.py) 를 읽어야하는데 node의 경로 (~catkin_ws/src/dynamic_reconfig_test/dynamic_reconfig_test.py)를 읽어서 빌드된 parameter 파일을 읽지못해서 발생하는 문제이다. 아래와 같이 라이브러리 경로를 직접 system path에 넣어주면 된다.

```
import sys
sys.path.append('/home/swyoo/usaywook/catkin_ws_for_vilab/devel/lib/python2.7/dist-packages/dynamic_reconfig_test')
from cfg import DynamicParamConfig
```



## 6. Execution

teminal 1

```
roscore
```

teminal 2

```
rosrun dynamic_reconfig_test dynamic_reconfig_test.py
```

terminal 3

```
rqt
```

Plugins -> Configuration -> Dynamic Reconfigure  

<figure>
  <center>
  <img src="https://enssionaut.com/files/attach/images/122/221/001/add411c4f76cca04860f86a7eb76b600.jpg" width="600">
  <figcaption>result as follows.</figcaption>
  </center>
</figure>

​                                          

## 7. using predefined values

how to set parameters with predefined values  

modify cfg file as follows:

```
#!/usr/bin/env python

PACKAGE = "dynamic_reconfig_test"

from dynamic_reconfigure.parameter_generator_catkin import *

gen = ParameterGenerator()

# gen.add("test_parameter", int_t, 0, "A test parameter", 0, 0, 100)
test_enum = gen.enum([
     gen.const("Low", int_t, 0, "A low value"),
     gen.const("Med", int_t, 50, "A medium vale"),
     gen.const("High", int_t, 100, "A high value")],
     "Test enum values")

gen.add("test_parameter", int_t, 0, "A test parameter", 0, 0, 100, edit_method=test_enum)

exit(gen.generate(PACKAGE, "dynamic_reconfig_test", "DynamicParam"))
```

build

```
cd ~/catkin_ws
carkin_make
```

execution

<figure>
  <center>
  <img src="https://enssionaut.com/files/attach/images/122/221/001/03de34625fc03674a8427229c2976bc4.jpg" width="600">
  <figcaption>result as follows.</figcaption>
  </center>
</figure>



## Reference

[c++패키지경우] https://enssionaut.com/board_robotics/1221



