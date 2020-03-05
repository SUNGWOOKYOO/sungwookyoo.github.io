# How to solve it?
Pycharm 에서 종종 debugging을 할때, `Collecting data...` 가 지속되다가 
`Unable to display frame variables` 가 뜨는 경우가 있다. 

이경우 간단히 해결할 수 있다.

`File > settings > Build, Execution, Deployment > Python Debugger >
Gevent compatible`
을 체크하면 디버거가 다시 잘 동작한다. 

