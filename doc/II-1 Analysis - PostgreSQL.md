
We store all included data in a PostgreSQL database. By and large we access this database using the `psycopg2` library.


```python
import psycopg2

from os import environ
conn = psycopg2.connect(dbname='postgres', user='postgres', host=environ['POSTGRES_1_PORT_5432_TCP_ADDR'])
cur = conn.cursor()
```


```python
cur.execute("SELECT COUNT(*) FROM people"); print(cur.fetchone())
cur.execute("SELECT COUNT(*) FROM action"); print(cur.fetchone())
```

    (189118,)
    (2695978,)



```python
conn.close()
```


```python
cd ~/work/data
```

    /home/jovyan/work/data



```python

```
