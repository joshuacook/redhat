import psycopg2
from os import environ
from numpy import array

keys = [
    #('people_id',False),
    ('act_id',False),
    ('act_date',False),
    ('act_category',True),
    ('act_char_1',True),
    ('act_char_2',True),
    ('act_char_3',True),
    ('act_char_4',True),
    ('act_char_5',True),
    ('act_char_6',True),
    ('act_char_7',True),
    ('act_char_8',True),
    ('act_char_9',True),
    ('act_char_10',False), 
    ('act_outcome',False), 
    ('people_id',False),
    ('ppl_char_1',True),
    ('ppl_group_1',False),
    ('ppl_char_2',True),
    ('ppl_date',False),
    ('ppl_char_3',True),
    ('ppl_char_4',True),
    ('ppl_char_5',True),
    ('ppl_char_6',True),
    ('ppl_char_7',True),
    ('ppl_char_8',True),
    ('ppl_char_9',True),
    ('ppl_char_10',False),
    ('ppl_char_11',False),
    ('ppl_char_12',False),
    ('ppl_char_13',False),
    ('ppl_char_14',False),
    ('ppl_char_15',False),
    ('ppl_char_16',False),
    ('ppl_char_17',False),
    ('ppl_char_18',False),
    ('ppl_char_19',False),
    ('ppl_char_20',False),
    ('ppl_char_21',False),
    ('ppl_char_22',False),
    ('ppl_char_23',False),
    ('ppl_char_24',False),
    ('ppl_char_25',False),
    ('ppl_char_26',False),
    ('ppl_char_27',False),
    ('ppl_char_28',False),
    ('ppl_char_29',False),
    ('ppl_char_30',False),
    ('ppl_char_31',False),
    ('ppl_char_32',False),
    ('ppl_char_33',False),
    ('ppl_char_34',False),
    ('ppl_char_35',False),
    ('ppl_char_36',False),
    ('ppl_char_37',False),
    ('ppl_char_38',False),
]


def one_hot_encode_row(action_id):

    conn = psycopg2.connect(dbname='postgres', user='postgres', host=environ['REDHAT_POSTGRES_1_PORT_5432_TCP_ADDR'])
    cur = conn.cursor()

    one_hot_join_query = """
    SELECT a.act_id, a.act_date, a.act_category,
            a.act_char_1, a.act_char_2, a.act_char_3,
            a.act_char_4, a.act_char_5, a.act_char_6,
            a.act_char_7, a.act_char_8, a.act_char_9,
            a.act_char_10, a.act_outcome, p.people_id,
            p.ppl_char_1, p.ppl_group_1, p.ppl_char_2,
            p.ppl_date, p.ppl_char_3, p.ppl_char_4,
            p.ppl_char_5, p.ppl_char_6, p.ppl_char_7,
            p.ppl_char_8, p.ppl_char_9, p.ppl_char_10,
            p.ppl_char_11, p.ppl_char_12, p.ppl_char_13,
            p.ppl_char_14, p.ppl_char_15, p.ppl_char_16,
            p.ppl_char_17, p.ppl_char_18, p.ppl_char_19,
            p.ppl_char_20, p.ppl_char_21, p.ppl_char_22,
            p.ppl_char_23, p.ppl_char_24, p.ppl_char_25,
            p.ppl_char_26, p.ppl_char_27, p.ppl_char_28,
            p.ppl_char_29, p.ppl_char_30, p.ppl_char_31,
            p.ppl_char_32, p.ppl_char_33, p.ppl_char_34,
            p.ppl_char_35, p.ppl_char_36, p.ppl_char_37,
            p.ppl_char_38
    FROM action a, people p 
    WHERE a.people_id = p.people_id AND a.act_id = '{}'
    """.format(action_id)

    cur.execute(one_hot_join_query)
    
    for ppl_act in cur.fetchall():
        print(ppl_act)
        cols = []
        vals = []
        for key, attr in zip(keys,ppl_act):
            if key[1]:
                if attr == None:
                    attr = 'none'
                col = str(key[0])+'_'+attr.replace(' ','_')
                attr = True
            else:
                col = key[0]
            cols.append(col)
            vals.append(str(attr))
        vals = "', '".join(vals)

        cols = ", ".join(cols)
        
        insert_sql = "INSERT into one_hot_ppl_act ({}) VALUES ('{}')".format(cols,vals)
        print(insert_sql)
        cur.execute(insert_sql)
        conn.commit()
        conn.close()
        return
                
def one_hot_from_table(cursor):
    one_hot_vec = (cursor.fetchone()[0]
     .replace('t','1')
     .replace('f','0')
     .replace('(','')
     .replace(')','')
     .split(',')
    )
    one_hot_vec = [float(el) for el in one_hot_vec]
    outcome = one_hot_vec[-1:][0]
    one_hot_vec = one_hot_vec[:-1]
    return array(one_hot_vec), outcome        
