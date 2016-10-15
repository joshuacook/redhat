import psycopg2
from os import environ
from numpy import array, frombuffer, zeros
from lib.helpers.one_hot_indices import one_hot_indices

def one_hot_encode_row(action_id):

    one_hot_vector = zeros(len(one_hot_indices)+1)
    
    conn = psycopg2.connect(dbname='postgres', user='postgres', host=environ['REDHAT_POSTGRES_1_PORT_5432_TCP_ADDR'])
    cur = conn.cursor()

    one_hot_join_query = """
    SELECT 
            a.act_char_1, a.act_char_2, a.act_char_3,
            a.act_char_4, a.act_char_5, a.act_char_6,
            a.act_char_7, a.act_char_8, a.act_char_9,
            a.act_char_10, 
            p.ppl_char_1, p.ppl_char_2,
            p.ppl_char_3, p.ppl_char_4,
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
    FROM action a INNER JOIN people p 
    ON a.people_id = p.people_id
    WHERE a.act_id = '{}'
    """.format(action_id)

    cols = ['act_char_1', 'act_char_2', 'act_char_3',
            'act_char_4', 'act_char_5', 'act_char_6',
            'act_char_7', 'act_char_8', 'act_char_9',
            'act_char_10', 'ppl_char_1', 'ppl_char_2',
            'ppl_char_3', 'ppl_char_4',
            'ppl_char_5', 'ppl_char_6', 'ppl_char_7',
            'ppl_char_8', 'ppl_char_9', 'ppl_char_10',
            'ppl_char_11', 'ppl_char_12', 'ppl_char_13',
            'ppl_char_14', 'ppl_char_15', 'ppl_char_16',
            'ppl_char_17', 'ppl_char_18', 'ppl_char_19',
            'ppl_char_20', 'ppl_char_21', 'ppl_char_22',
            'ppl_char_23', 'ppl_char_24', 'ppl_char_25',
            'ppl_char_26', 'ppl_char_27', 'ppl_char_28',
            'ppl_char_29', 'ppl_char_30', 'ppl_char_31',
            'ppl_char_32', 'ppl_char_33', 'ppl_char_34',
            'ppl_char_35', 'ppl_char_36', 'ppl_char_37']
    cur.execute(one_hot_join_query)
    
    res = list(cur.fetchall()[0])
    ppl_char_38 = res.pop()
    labels = [str(col)+'_'+str(re).replace(' ','_') for col,re in zip(cols,res)]
    indices = [one_hot_indices[label] for label in labels]

    for index in indices:
        one_hot_vector[index] = 1
    one_hot_vector[-1] = ppl_char_38

    update_sql = """
    UPDATE action
    SET act_one_hot_encoding = {}
    WHERE act_id='{}'
    """.format(psycopg2.Binary(one_hot_vector), action_id)
    cur.execute(update_sql)
    conn.commit()
    conn.close()

    return one_hot_vector
                
def one_hot_from_table(action_id):
    
    conn = psycopg2.connect(dbname='postgres', user='postgres', host=environ['REDHAT_POSTGRES_1_PORT_5432_TCP_ADDR'])
    cur = conn.cursor()
    cur.execute("""
    SELECT act_one_hot_encoding
    FROM action
    WHERE act_id = '{}'
    """.format(action_id))
    buf = cur.fetchone()[0]
    cur.close()
    return frombuffer(buf)

def one_hot_and_outcome(action_id):
    conn = psycopg2.connect(dbname='postgres', user='postgres', host=environ['REDHAT_POSTGRES_1_PORT_5432_TCP_ADDR'])
    cur = conn.cursor()
    cur.execute("""
    SELECT act_outcome
    FROM action
    WHERE act_id = '{}';
    """.format(action_id))
    outcome = cur.fetchone()[0]
    cur.close()
    return one_hot_from_table(action_id), int(outcome)

def pull_actions(limit=1000, action_type='one-hot'):
    conn = psycopg2.connect(dbname='postgres', user='postgres', host=environ['REDHAT_POSTGRES_1_PORT_5432_TCP_ADDR'])
    cur = conn.cursor()
    if action_type=='one-hot':
        sql = """
            SELECT act_id FROM action 
            WHERE act_outcome = True OR act_outcome = False 
            AND act_one_hot_encoding is NULL 
            LIMIT {};""".format(limit)
    elif action_type == 'training':
        sql = """
            SELECT act_id FROM action 
            WHERE act_one_hot_encoding is not NULL 
            LIMIT {};""".format(limit)
    elif action_type == 'testing':
        sql = """
            SELECT act_id FROM action 
            WHERE act_one_hot_encoding is not NULL 
            AND act_outcome is NULL
            LIMIT {};""".format(limit)
    cur.execute(sql)
    action_ids = [action_id[0] for action_id in cur.fetchall()]
    return action_ids
        
def pull_actions_and_one_hot_encode():    
    for action_id in pull_actions():
        one_hot_encode_row(action_id)
    
