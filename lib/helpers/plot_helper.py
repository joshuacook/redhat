import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lib.helpers.database_helper import connect_to_postgres

def bar_plot(col,table):
    vals,labels = hist_buckets(col,table)
    x = np.arange(len(vals))
    y = np.array(vals)
    f = plt.figure(figsize=(12,3))
    ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
    sns.barplot(x=x, y=y,palette='Greens_d')
    ax.set_title("Counts for {} in {}".format(col,table))
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace('type ','') for label in labels])

def hist_buckets(column, table):
    
    conn, cur = connect_to_postgres()

    sql = "SELECT DISTINCT {} FROM {};".format(column,table)
    cur.execute(sql)

    labels = [str(l[0]) for l in cur.fetchall()]
    labels.sort()
    sql = "SELECT {} FROM ".format(','.join(labels).replace(' ','_'))
    sql_rows = ["(SELECT COUNT({}) FROM {} WHERE {} = '{}') as {}".format(column,table,column,label,label.replace(' ','_')) for label in labels]

    sql += ",".join(sql_rows)
    
    cur.execute(sql)
    bins = cur.fetchall()[0]
    bins = [int(bn.replace('(','').replace(')','')) for bn in bins]

    conn.close()

    return bins, labels
