CREATE TABLE people (
    people_id TEXT PRIMARY KEY,
    ppl_char_1 TEXT,
    ppl_group_1 TEXT,
    ppl_char_2 TEXT,
    ppl_date TIMESTAMP,
    ppl_char_3 TEXT,
    ppl_char_4 TEXT,
    ppl_char_5 TEXT,
    ppl_char_6 TEXT,
    ppl_char_7 TEXT,
    ppl_char_8 TEXT,
    ppl_char_9 TEXT,
    ppl_char_10 BOOLEAN,
    ppl_char_11 BOOLEAN,
    ppl_char_12 BOOLEAN,
    ppl_char_13 BOOLEAN,
    ppl_char_14 BOOLEAN,
    ppl_char_15 BOOLEAN,
    ppl_char_16 BOOLEAN,
    ppl_char_17 BOOLEAN,
    ppl_char_18 BOOLEAN,
    ppl_char_19 BOOLEAN,
    ppl_char_20 BOOLEAN,
    ppl_char_21 BOOLEAN,
    ppl_char_22 BOOLEAN,
    ppl_char_23 BOOLEAN,
    ppl_char_24 BOOLEAN,
    ppl_char_25 BOOLEAN,
    ppl_char_26 BOOLEAN,
    ppl_char_27 BOOLEAN,
    ppl_char_28 BOOLEAN,
    ppl_char_29 BOOLEAN,
    ppl_char_30 BOOLEAN,
    ppl_char_31 BOOLEAN,
    ppl_char_32 BOOLEAN,
    ppl_char_33 BOOLEAN,
    ppl_char_34 BOOLEAN,
    ppl_char_35 BOOLEAN,
    ppl_char_36 BOOLEAN,
    ppl_char_37 BOOLEAN,
    ppl_char_38 REAL
);
COPY people FROM '/docker-entrypoint-init.d/people.csv' HEADER DELIMITER ',' CSV; 
CREATE TABLE action (
    people_id TEXT REFERENCES people,
    act_id TEXT PRIMARY KEY,
    act_date TIMESTAMP, 
    act_category TEXT,
    act_char_1 TEXT,
    act_char_2 TEXT,
    act_char_3 TEXT,
    act_char_4 TEXT,
    act_char_5 TEXT,
    act_char_6 TEXT,
    act_char_7 TEXT,
    act_char_8 TEXT,
    act_char_9 TEXT,
    act_char_10 TEXT,
    act_outcome BOOLEAN DEFAULT NULL,
    act_one_hot_encoding BYTEA
);
COPY action FROM '/docker-entrypoint-init.d/act_test.csv' HEADER DELIMITER ',' CSV; 
COPY action FROM '/docker-entrypoint-init.d/act_train_1.csv' HEADER DELIMITER ',' CSV; 
COPY action FROM '/docker-entrypoint-init.d/act_train_2.csv' HEADER DELIMITER ',' CSV; 
COPY action FROM '/docker-entrypoint-init.d/act_train_3.csv' HEADER DELIMITER ',' CSV; 
