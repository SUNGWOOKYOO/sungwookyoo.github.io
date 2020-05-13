# 작성시 유의사항 정리
typora, jupyter notebook, jekyll 모두에서 호환이 되도록 작성하고싶다.
따옴표가 있으면 꼭 붙혀줄주어야한다. 

## 그림 
크기조절을 하기 위해서는 다음과 같이 작성하는 것이 좋다.
github.io pages에 rendering 하려면 반드시 git add를 해야한다.
<img src="URL" width="600"> 
githubio 페이지에 렌더링 했을떄, 약 600정도사이즈로 하면 적당하다.

## 텍스트
### color
<span style="color:red"> TEXT </span>

### footnote
jupyter notebook에서는 보이지 않지만, jekyll에 rendering하면 보인다. 
어떤 글[^1] 
...
[1]: url "description"


## 수식
$\mathbb{R}$ 실수표현
$\mathcal{V}$ 집합표현
$\vert a  \verl$ abs(a)

