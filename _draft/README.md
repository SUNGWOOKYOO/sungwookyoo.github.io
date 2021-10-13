# 작성시 유의사항 정리
typora, jupyter notebook, jekyll 모두에서 호환이 되도록 작성하고싶다.
따옴표가 있으면 꼭 붙혀줄주어야한다. 

## 그림 
크기조절을 하기 위해서는 다음과 같이 작성하는 것이 좋다.
github.io pages에 rendering 하려면 반드시 git add를 해야한다.
```
<img src="URL" width="600"> 
```
githubio 페이지에 렌더링 했을떄, 약 600정도사이즈로 하면 적당하다.

### 심화

가운데 정렬을 하고, 캡션도 넣고 싶을 때
```
<center><figure>
  <img src="address" width="%50" style="background-color:white"; title="#">
  <figcaption> Fig1. </figcaption>
</figure></center>
```

## 텍스트
### color
`<span style="color:red"> TEXT </span>`

### footnote
jupyter notebook에서는 보이지 않지만, jekyll에 rendering하면 보인다. 
어떤 글`[^1]` 
...
```
[1]: url "description"
```

## 수식
$\mathbb{R}$ 실수표현
$\mathcal{V}$ 집합표현
$\vert a  \verl$ abs(a)  

inline 수식에 대해서 | 을 escape 기호를 사용하여  \\| 로 작성해 주어야 블로그에서 안깨짐

inline 수식에 대해서 * 을 escape 기호를 사용하여  \\* 로 작성해 주어야 블로그에서 안깨짐

inline 수식에 대해서 _ 을 escape 기호를 사용하여  \\_ 로 작성해 주어야 블로그에서 안깨짐

\cancel{}을 사용하기위해서 \require{cancel}을 선언해 주어야함

\textcolor{red}{}를 사용하기위해서 \require{color}를 선언해 주어야함


## HTML
### picture
```
<center>
<figure>
  <img src="address" width="300" style="background-color:white"; title="#">
  <figcaption> Fig1. </figcaption>
</figure>
</center>
```

jupyter의 css 를 따르되, two column 이미지를 만들고 싶을 경우

```html
<figure style="text-align: center;">
    <img src="/assets/images/Word2Vec_files/skipgram.png" style="background-color:white;width:300px;height:300px;"> <br>
    <figcaption style="text-align: center;"> <b>그림 1-2</b>. Skip-Gram </figcaption>
</figure>
```

## 텍스트
### color
<span style="color:red"> TEXT </span>

### footnote
jupyter notebook에서는 보이지 않지만, jekyll에 rendering하면 보인다. 
어떤 글[^1] 
...

[^1]: url "description"

### link

첨자 표현 <sub>[1]</sub>

This is a example of footnote[<sup>[1]</sup>](#fn1). Another footnote[<sup>[2]</sup>](#fn2).

<span id="fn1"> [footnote 1]</span>: url ""
<span id="fn2"> [footnote 2]</span>: url ""

### conceal
```
<details> <summary> </summary>
내용이 들어간다.
</details>
```

### bullet
점 불렛
```
<ul>
  <li> </ul>
  <li> </ul>
</ul>
```

숫자 불렛
```
<ol>
  <li> </ul>
  <li> </ul>
</ol>
```

## 수식
$\mathbb{R}$ 실수표현
$\mathcal{V}$ 집합표현
$\vert a  \vert$ abs(a)  

inline 수식에 대해서 | 을 escape 기호를 사용하여  \mid 혹은 \vert 로 작성해 주어야 블로그에서 안깨짐

inline 수식에 대해서 * 을 escape 기호를 사용하여  \\* 로 작성해 주어야 블로그에서 안깨짐

inline 수식에 대해서 _ 을 escape 기호를 사용하여  \\_ 로 작성해 주어야 블로그에서 안깨짐

\cancel{}을 사용하기위해서 \require{cancel}을 선언해 주어야함

\textcolor{red}{}를 사용하기위해서 \require{color}를 선언해 주어야함

\underbrace{expression1}_{expression2} 를 사용하여 아래 중괄호를 만든다.


## minimal mistakes posting
### gallery, video
follow [this tutorial](https://mmistakes.github.io/minimal-mistakes/docs/helpers/#gallery)
