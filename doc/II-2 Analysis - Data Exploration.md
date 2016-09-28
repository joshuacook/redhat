

```python
cd /home/jovyan/work/data
```

    /home/jovyan/work/data



```python
from os import chdir, environ
import numpy as np
import pandas as pd
import psycopg2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
action_train_data = pd.read_csv("act_train_head.csv")
action_test_data = pd.read_csv("act_test_head.csv")
people_data = pd.read_csv("people_head.csv")
```


```python
action_train_data.head().to_latex()
```




    '\\begin{tabular}{lllllrrrrrrrrrlr}\n\\toprule\n{} & people\\_id &   activity\\_id &        date & activity\\_category &  char\\_1 &  char\\_2 &  char\\_3 &  char\\_4 &  char\\_5 &  char\\_6 &  char\\_7 &  char\\_8 &  char\\_9 &  char\\_10 &  outcome \\\\\n\\midrule\n0 &   ppl\\_100 &  act2\\_1734928 &  2023-08-26 &            type 4 &     NaN &     NaN &     NaN &     NaN &     NaN &     NaN &     NaN &     NaN &     NaN &  type 76 &        0 \\\\\n1 &   ppl\\_100 &  act2\\_2434093 &  2022-09-27 &            type 2 &     NaN &     NaN &     NaN &     NaN &     NaN &     NaN &     NaN &     NaN &     NaN &   type 1 &        0 \\\\\n2 &   ppl\\_100 &  act2\\_3404049 &  2022-09-27 &            type 2 &     NaN &     NaN &     NaN &     NaN &     NaN &     NaN &     NaN &     NaN &     NaN &   type 1 &        0 \\\\\n3 &   ppl\\_100 &  act2\\_3651215 &  2023-08-04 &            type 2 &     NaN &     NaN &     NaN &     NaN &     NaN &     NaN &     NaN &     NaN &     NaN &   type 1 &        0 \\\\\n4 &   ppl\\_100 &  act2\\_4109017 &  2023-08-26 &            type 2 &     NaN &     NaN &     NaN &     NaN &     NaN &     NaN &     NaN &     NaN &     NaN &   type 1 &        0 \\\\\n\\bottomrule\n\\end{tabular}\n'




```python
action_test_data.head().to_latex()
```




    '\\begin{tabular}{lllllllllllllll}\n\\toprule\n{} &   people\\_id &   activity\\_id &        date & activity\\_category &   char\\_1 &   char\\_2 &  char\\_3 &  char\\_4 &  char\\_5 &  char\\_6 &  char\\_7 &   char\\_8 &   char\\_9 &    char\\_10 \\\\\n\\midrule\n0 &  ppl\\_100004 &   act1\\_249281 &  2022-07-20 &            type 1 &   type 5 &  type 10 &  type 5 &  type 1 &  type 6 &  type 1 &  type 1 &   type 7 &   type 4 &        NaN \\\\\n1 &  ppl\\_100004 &   act2\\_230855 &  2022-07-20 &            type 5 &      NaN &      NaN &     NaN &     NaN &     NaN &     NaN &     NaN &      NaN &      NaN &   type 682 \\\\\n2 &   ppl\\_10001 &   act1\\_240724 &  2022-10-14 &            type 1 &  type 12 &   type 1 &  type 5 &  type 4 &  type 6 &  type 1 &  type 1 &  type 13 &  type 10 &        NaN \\\\\n3 &   ppl\\_10001 &    act1\\_83552 &  2022-11-27 &            type 1 &  type 20 &  type 10 &  type 5 &  type 4 &  type 6 &  type 1 &  type 1 &   type 5 &   type 5 &        NaN \\\\\n4 &   ppl\\_10001 &  act2\\_1043301 &  2022-10-15 &            type 5 &      NaN &      NaN &     NaN &     NaN &     NaN &     NaN &     NaN &      NaN &      NaN &  type 3015 \\\\\n\\bottomrule\n\\end{tabular}\n'




```python
people_data.head().to_latex()
```




    '\\begin{tabular}{lllllllllllllllllllllllllllllllllllllllllr}\n\\toprule\n{} &   people\\_id &  char\\_1 &      group\\_1 &  char\\_2 &        date &   char\\_3 &   char\\_4 &  char\\_5 &  char\\_6 &   char\\_7 &  char\\_8 &  char\\_9 & char\\_10 & char\\_11 & char\\_12 & char\\_13 & char\\_14 & char\\_15 & char\\_16 & char\\_17 & char\\_18 & char\\_19 & char\\_20 & char\\_21 & char\\_22 & char\\_23 & char\\_24 & char\\_25 & char\\_26 & char\\_27 & char\\_28 & char\\_29 & char\\_30 & char\\_31 & char\\_32 & char\\_33 & char\\_34 & char\\_35 & char\\_36 & char\\_37 &  char\\_38 \\\\\n\\midrule\n0 &     ppl\\_100 &  type 2 &  group 17304 &  type 2 &  2021-06-29 &   type 5 &   type 5 &  type 5 &  type 3 &  type 11 &  type 2 &  type 2 &    True &   False &   False &    True &    True &   False &    True &   False &   False &   False &   False &    True &   False &   False &   False &   False &   False &    True &    True &   False &    True &    True &   False &   False &    True &    True &    True &   False &       36 \\\\\n1 &  ppl\\_100002 &  type 2 &   group 8688 &  type 3 &  2021-01-06 &  type 28 &   type 9 &  type 5 &  type 3 &  type 11 &  type 2 &  type 4 &   False &   False &    True &    True &   False &   False &   False &    True &   False &   False &   False &   False &   False &    True &   False &    True &    True &    True &   False &   False &    True &    True &    True &    True &    True &    True &    True &   False &       76 \\\\\n2 &  ppl\\_100003 &  type 2 &  group 33592 &  type 3 &  2022-06-10 &   type 4 &   type 8 &  type 5 &  type 2 &   type 5 &  type 2 &  type 2 &    True &    True &    True &    True &    True &    True &   False &    True &   False &    True &   False &    True &    True &    True &    True &    True &    True &    True &    True &   False &   False &    True &    True &    True &    True &   False &    True &    True &       99 \\\\\n3 &  ppl\\_100004 &  type 2 &  group 22593 &  type 3 &  2022-07-20 &  type 40 &  type 25 &  type 9 &  type 4 &  type 16 &  type 2 &  type 2 &    True &    True &    True &    True &    True &   False &    True &    True &    True &    True &    True &    True &    True &    True &   False &    True &    True &    True &    True &    True &    True &    True &    True &    True &    True &    True &    True &    True &       76 \\\\\n4 &  ppl\\_100006 &  type 2 &   group 6534 &  type 3 &  2022-07-27 &  type 40 &  type 25 &  type 9 &  type 3 &   type 8 &  type 2 &  type 2 &   False &   False &   False &   False &   False &   False &   False &   False &   False &   False &   False &   False &   False &   False &   False &   False &   False &   False &   False &   False &   False &    True &   False &   False &   False &    True &    True &   False &       84 \\\\\n\\bottomrule\n\\end{tabular}\n'




```python

```
