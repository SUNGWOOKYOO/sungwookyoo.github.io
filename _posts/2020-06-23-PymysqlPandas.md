---
title: "Pandas DataFrame to MySQL"
excerpt: "using MySQL and pymysql, data preprocessing can be easily possible."
categories:
 - study
 - tips
tags:
 - database
 - python
 - mysql
use_math: true
last_modified_at: "2020-06-23"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/teaser.jpg
 overlay_filter: 0.5
---

<div class="prompt input_prompt">
In&nbsp;[14]:
</div>

<div class="input_area" markdown="1">

```python
import pymysql, argparse, os, sys, json
from sqlalchemy import create_engine
import pandas as pd
```

</div>

# How to use pymysql using Pandas, MySQL

## Step 0. Connect to MySQL

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
parser = argparse.ArgumentParser()
parser.add_argument('-user', help="mysql database user", type=str, required=False, default='swyoo')
parser.add_argument('-pw', help="password", type=str, required=False, default='****')
parser.add_argument('-host', help="ip address", type=str, required=False, default='***.***.***.***')
parser.add_argument('-db', help="database name", type=str, required=False, default='classicmodels')
parser.add_argument('-charset', help="character set to use", type=str, required=False, default='utf8mb4')
sys.argv = ['-f']
args = parser.parse_args()
args
```

</div>




{:.output_data_text}

```
Namespace(charset='utf8mb4', db='classicmodels', host='125.191.6.186', pw='1360', user='swyoo')
```



<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
# helper functions 
sql = lambda command: pd.read_sql(command, con)
def fetch(command):
    cursor.execute(command)
    return cursor.fetchall()
```

</div>

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
con = pymysql.connect(host=args.host, user=args.user, password=args.pw, use_unicode=True, charset=args.charset)
cursor = con.cursor()
```

</div>

## Step 1. Use sqlalchemy, create engine
I use sample database `classicmodels` on MySQL in [this tutorial](https://www.mysqltutorial.org/mysql-sample-database.aspx/) 

Create engine to convert pandas.Datafame to a table in MySQL. <br>
Therefore, connect to a database `classicmodels` on MySQL 

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
fetch("create database if not exists testdb")

""" insert dataset to database """
# db_data = 'mysql+pymysql://' + '<USER-NAME>' + ':' + '<PASSWORD>' + '@' + '***.***.***.***' + ':3306/' + '<DB-NAME>' + '?charset=utf8mb4'
db_data = "mysql+pymysql://{}:{}@{}:3306/{}?charset={}".format(args.user, args.pw, args.host, args.db, args.charset)
engine = create_engine(db_data).connect()
```

</div>

{:.output_stream}

```
/home/swyoo/anaconda3/envs/torch/lib/python3.7/site-packages/pymysql/cursors.py:170: Warning: (1007, "Can't create database 'testdb'; database exists")
  result = self._query(query)

```

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
fetch("use classicmodels")
sql("show tables")
```

</div>




<div markdown="0">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tables_in_classicmodels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>customers</td>
    </tr>
    <tr>
      <th>1</th>
      <td>employees</td>
    </tr>
    <tr>
      <th>2</th>
      <td>offices</td>
    </tr>
    <tr>
      <th>3</th>
      <td>orderdetails</td>
    </tr>
    <tr>
      <th>4</th>
      <td>orders</td>
    </tr>
    <tr>
      <th>5</th>
      <td>payments</td>
    </tr>
    <tr>
      <th>6</th>
      <td>productlines</td>
    </tr>
    <tr>
      <th>7</th>
      <td>products</td>
    </tr>
  </tbody>
</table>
</div>
</div>



## Step2. Prepare dataframe

<div class="prompt input_prompt">
In&nbsp;[11]:
</div>

<div class="input_area" markdown="1">

```python
!wget -N http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz
!gzip -d reviews_Musical_Instruments_5.json.gz    
```

</div>

{:.output_stream}

```
--2020-05-08 18:03:39--  http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz
Resolving snap.stanford.edu (snap.stanford.edu)... 171.64.75.80
Connecting to snap.stanford.edu (snap.stanford.edu)|171.64.75.80|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 2460495 (2.3M) [application/x-gzip]
Saving to: ‘reviews_Musical_Instruments_5.json.gz’

reviews_Musical_Ins 100%[===================>]   2.35M  4.85MB/s    in 0.5s    

2020-05-08 18:03:40 (4.85 MB/s) - ‘reviews_Musical_Instruments_5.json.gz’ saved [2460495/2460495]


```

<div class="prompt input_prompt">
In&nbsp;[17]:
</div>

<div class="input_area" markdown="1">

```python
DATA_JSON = "reviews_Musical_Instruments_5.json"
""" read data from *.json """
users_id = []
items_id = []
ratings = []
reviews = []
with open(DATA_JSON, 'r') as f:
    for line in f:
        js = json.loads(line)
        if str(js['reviewerID']) == 'unknown':
            print("unknown")
            continue
        if str(js['asin']) == "unknown":
            print("unknown")
            continue
        reviews.append(js['reviewText'])
        users_id.append(str(js['reviewerID']) + ",")
        items_id.append(str(js['asin']) + ",")
        ratings.append(str(js['overall']))

df = pd.DataFrame(
    {'user_id': pd.Series(users_id),
     'item_id': pd.Series(items_id),
     'ratings': pd.Series(ratings),
     'reviews': pd.Series(reviews)}
)[['user_id', 'item_id', 'ratings', 'reviews']]

"""
https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html
user_id column을 pandas.Series.astype 중에서 dtype=category 데이터 타입으로 바꾼후
"""
user_id_int = df.user_id.astype('category').cat.codes
item_id_int = df.item_id.astype('category').cat.codes
df.user_id = user_id_int.astype('int64')
df.item_id = item_id_int.astype('int64')
df.ratings = df.ratings.astype('float64')
```

</div>

<div class="prompt input_prompt">
In&nbsp;[18]:
</div>

<div class="input_area" markdown="1">

```python
df
```

</div>




<div markdown="0">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>item_id</th>
      <th>ratings</th>
      <th>reviews</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>550</td>
      <td>0</td>
      <td>5.0</td>
      <td>Not much to write about here, but it does exac...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55</td>
      <td>0</td>
      <td>5.0</td>
      <td>The product does exactly as it should and is q...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>92</td>
      <td>0</td>
      <td>5.0</td>
      <td>The primary job of this device is to block the...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>482</td>
      <td>0</td>
      <td>5.0</td>
      <td>Nice windscreen protects my MXL mic and preven...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1145</td>
      <td>0</td>
      <td>5.0</td>
      <td>This pop filter is great. It looks and perform...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10256</th>
      <td>45</td>
      <td>899</td>
      <td>5.0</td>
      <td>Great, just as expected.  Thank to all.</td>
    </tr>
    <tr>
      <th>10257</th>
      <td>279</td>
      <td>899</td>
      <td>5.0</td>
      <td>I've been thinking about trying the Nanoweb st...</td>
    </tr>
    <tr>
      <th>10258</th>
      <td>1390</td>
      <td>899</td>
      <td>4.0</td>
      <td>I have tried coated strings in the past ( incl...</td>
    </tr>
    <tr>
      <th>10259</th>
      <td>720</td>
      <td>899</td>
      <td>4.0</td>
      <td>Well, MADE by Elixir and DEVELOPED with Taylor...</td>
    </tr>
    <tr>
      <th>10260</th>
      <td>683</td>
      <td>899</td>
      <td>4.0</td>
      <td>These strings are really quite good, but I wou...</td>
    </tr>
  </tbody>
</table>
<p>10261 rows × 4 columns</p>
</div>
</div>



## Step3. pandas to mysql table using engine

<div class="prompt input_prompt">
In&nbsp;[20]:
</div>

<div class="input_area" markdown="1">

```python
df.to_sql(name='music', con=engine, if_exists='replace')
print("insert all data set to database done")
```

</div>

{:.output_stream}

```
insert all data set to database done

```

<div class="prompt input_prompt">
In&nbsp;[21]:
</div>

<div class="input_area" markdown="1">

```python
sql("select * from music")
```

</div>




<div markdown="0">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>user_id</th>
      <th>item_id</th>
      <th>ratings</th>
      <th>reviews</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>550</td>
      <td>0</td>
      <td>5.0</td>
      <td>Not much to write about here, but it does exac...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>55</td>
      <td>0</td>
      <td>5.0</td>
      <td>The product does exactly as it should and is q...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>92</td>
      <td>0</td>
      <td>5.0</td>
      <td>The primary job of this device is to block the...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>482</td>
      <td>0</td>
      <td>5.0</td>
      <td>Nice windscreen protects my MXL mic and preven...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1145</td>
      <td>0</td>
      <td>5.0</td>
      <td>This pop filter is great. It looks and perform...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10256</th>
      <td>10256</td>
      <td>45</td>
      <td>899</td>
      <td>5.0</td>
      <td>Great, just as expected.  Thank to all.</td>
    </tr>
    <tr>
      <th>10257</th>
      <td>10257</td>
      <td>279</td>
      <td>899</td>
      <td>5.0</td>
      <td>I've been thinking about trying the Nanoweb st...</td>
    </tr>
    <tr>
      <th>10258</th>
      <td>10258</td>
      <td>1390</td>
      <td>899</td>
      <td>4.0</td>
      <td>I have tried coated strings in the past ( incl...</td>
    </tr>
    <tr>
      <th>10259</th>
      <td>10259</td>
      <td>720</td>
      <td>899</td>
      <td>4.0</td>
      <td>Well, MADE by Elixir and DEVELOPED with Taylor...</td>
    </tr>
    <tr>
      <th>10260</th>
      <td>10260</td>
      <td>683</td>
      <td>899</td>
      <td>4.0</td>
      <td>These strings are really quite good, but I wou...</td>
    </tr>
  </tbody>
</table>
<p>10261 rows × 5 columns</p>
</div>
</div>


