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
    act_outcome BOOLEAN DEFAULT NULL
);
COPY action FROM '/docker-entrypoint-init.d/act_test.csv' HEADER DELIMITER ',' CSV; 
COPY action FROM '/docker-entrypoint-init.d/act_train.csv' HEADER DELIMITER ',' CSV; 
CREATE TABLE one_hot_ppl_act (
    people_id TEXT,
    ppl_group_1 TEXT,
    ppl_date TIMESTAMP,
    act_date TIMESTAMP,
    act_id TEXT,
    act_outcome BOOLEAN,
    ppl_char_1_type_1 BOOLEAN DEFAULT FALSE,
    ppl_char_1_type_2 BOOLEAN DEFAULT FALSE,
    ppl_char_2_type_1 BOOLEAN DEFAULT FALSE,
    ppl_char_2_type_2 BOOLEAN DEFAULT FALSE,
    ppl_char_2_type_3 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_1 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_10 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_11 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_12 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_13 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_14 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_15 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_16 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_17 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_18 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_19 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_2 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_20 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_21 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_22 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_23 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_24 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_25 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_26 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_27 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_28 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_29 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_3 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_30 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_31 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_32 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_33 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_34 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_35 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_36 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_37 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_38 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_39 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_4 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_40 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_41 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_42 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_44 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_5 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_6 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_7 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_8 BOOLEAN DEFAULT FALSE,
    ppl_char_3_type_9 BOOLEAN DEFAULT FALSE,
    ppl_char_4_type_1 BOOLEAN DEFAULT FALSE,
    ppl_char_4_type_10 BOOLEAN DEFAULT FALSE,
    ppl_char_4_type_11 BOOLEAN DEFAULT FALSE,
    ppl_char_4_type_12 BOOLEAN DEFAULT FALSE,
    ppl_char_4_type_13 BOOLEAN DEFAULT FALSE,
    ppl_char_4_type_14 BOOLEAN DEFAULT FALSE,
    ppl_char_4_type_15 BOOLEAN DEFAULT FALSE,
    ppl_char_4_type_16 BOOLEAN DEFAULT FALSE,
    ppl_char_4_type_17 BOOLEAN DEFAULT FALSE,
    ppl_char_4_type_18 BOOLEAN DEFAULT FALSE,
    ppl_char_4_type_19 BOOLEAN DEFAULT FALSE,
    ppl_char_4_type_2 BOOLEAN DEFAULT FALSE,
    ppl_char_4_type_20 BOOLEAN DEFAULT FALSE,
    ppl_char_4_type_21 BOOLEAN DEFAULT FALSE,
    ppl_char_4_type_22 BOOLEAN DEFAULT FALSE,
    ppl_char_4_type_23 BOOLEAN DEFAULT FALSE,
    ppl_char_4_type_24 BOOLEAN DEFAULT FALSE,
    ppl_char_4_type_25 BOOLEAN DEFAULT FALSE,
    ppl_char_4_type_3 BOOLEAN DEFAULT FALSE,
    ppl_char_4_type_4 BOOLEAN DEFAULT FALSE,
    ppl_char_4_type_5 BOOLEAN DEFAULT FALSE,
    ppl_char_4_type_6 BOOLEAN DEFAULT FALSE,
    ppl_char_4_type_7 BOOLEAN DEFAULT FALSE,
    ppl_char_4_type_8 BOOLEAN DEFAULT FALSE,
    ppl_char_4_type_9 BOOLEAN DEFAULT FALSE,
    ppl_char_5_type_1 BOOLEAN DEFAULT FALSE,
    ppl_char_5_type_2 BOOLEAN DEFAULT FALSE,
    ppl_char_5_type_3 BOOLEAN DEFAULT FALSE,
    ppl_char_5_type_4 BOOLEAN DEFAULT FALSE,
    ppl_char_5_type_5 BOOLEAN DEFAULT FALSE,
    ppl_char_5_type_6 BOOLEAN DEFAULT FALSE,
    ppl_char_5_type_7 BOOLEAN DEFAULT FALSE,
    ppl_char_5_type_8 BOOLEAN DEFAULT FALSE,
    ppl_char_5_type_9 BOOLEAN DEFAULT FALSE,
    ppl_char_6_type_1 BOOLEAN DEFAULT FALSE,
    ppl_char_6_type_2 BOOLEAN DEFAULT FALSE,
    ppl_char_6_type_3 BOOLEAN DEFAULT FALSE,
    ppl_char_6_type_4 BOOLEAN DEFAULT FALSE,
    ppl_char_6_type_5 BOOLEAN DEFAULT FALSE,
    ppl_char_6_type_6 BOOLEAN DEFAULT FALSE,
    ppl_char_6_type_7 BOOLEAN DEFAULT FALSE,
    ppl_char_7_type_1 BOOLEAN DEFAULT FALSE,
    ppl_char_7_type_10 BOOLEAN DEFAULT FALSE,
    ppl_char_7_type_11 BOOLEAN DEFAULT FALSE,
    ppl_char_7_type_12 BOOLEAN DEFAULT FALSE,
    ppl_char_7_type_13 BOOLEAN DEFAULT FALSE,
    ppl_char_7_type_14 BOOLEAN DEFAULT FALSE,
    ppl_char_7_type_15 BOOLEAN DEFAULT FALSE,
    ppl_char_7_type_16 BOOLEAN DEFAULT FALSE,
    ppl_char_7_type_17 BOOLEAN DEFAULT FALSE,
    ppl_char_7_type_18 BOOLEAN DEFAULT FALSE,
    ppl_char_7_type_19 BOOLEAN DEFAULT FALSE,
    ppl_char_7_type_2 BOOLEAN DEFAULT FALSE,
    ppl_char_7_type_20 BOOLEAN DEFAULT FALSE,
    ppl_char_7_type_21 BOOLEAN DEFAULT FALSE,
    ppl_char_7_type_22 BOOLEAN DEFAULT FALSE,
    ppl_char_7_type_23 BOOLEAN DEFAULT FALSE,
    ppl_char_7_type_24 BOOLEAN DEFAULT FALSE,
    ppl_char_7_type_25 BOOLEAN DEFAULT FALSE,
    ppl_char_7_type_3 BOOLEAN DEFAULT FALSE,
    ppl_char_7_type_4 BOOLEAN DEFAULT FALSE,
    ppl_char_7_type_5 BOOLEAN DEFAULT FALSE,
    ppl_char_7_type_6 BOOLEAN DEFAULT FALSE,
    ppl_char_7_type_7 BOOLEAN DEFAULT FALSE,
    ppl_char_7_type_8 BOOLEAN DEFAULT FALSE,
    ppl_char_7_type_9 BOOLEAN DEFAULT FALSE,
    ppl_char_8_type_1 BOOLEAN DEFAULT FALSE,
    ppl_char_8_type_2 BOOLEAN DEFAULT FALSE,
    ppl_char_8_type_3 BOOLEAN DEFAULT FALSE,
    ppl_char_8_type_4 BOOLEAN DEFAULT FALSE,
    ppl_char_8_type_5 BOOLEAN DEFAULT FALSE,
    ppl_char_8_type_6 BOOLEAN DEFAULT FALSE,
    ppl_char_8_type_7 BOOLEAN DEFAULT FALSE,
    ppl_char_8_type_8 BOOLEAN DEFAULT FALSE,
    ppl_char_9_type_1 BOOLEAN DEFAULT FALSE,
    ppl_char_9_type_2 BOOLEAN DEFAULT FALSE,
    ppl_char_9_type_3 BOOLEAN DEFAULT FALSE,
    ppl_char_9_type_4 BOOLEAN DEFAULT FALSE,
    ppl_char_9_type_5 BOOLEAN DEFAULT FALSE,
    ppl_char_9_type_6 BOOLEAN DEFAULT FALSE,
    ppl_char_9_type_7 BOOLEAN DEFAULT FALSE,
    ppl_char_9_type_8 BOOLEAN DEFAULT FALSE,
    ppl_char_9_type_9 BOOLEAN DEFAULT FALSE,
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
    ppl_char_38 REAL,
    act_category_type_1 BOOLEAN DEFAULT FALSE,
    act_category_type_2 BOOLEAN DEFAULT FALSE,
    act_category_type_3 BOOLEAN DEFAULT FALSE,
    act_category_type_4 BOOLEAN DEFAULT FALSE,
    act_category_type_5 BOOLEAN DEFAULT FALSE,
    act_category_type_6 BOOLEAN DEFAULT FALSE,
    act_category_type_7 BOOLEAN DEFAULT FALSE,
    act_char_1_none BOOLEAN DEFAULT FALSE,
    act_char_1_type_1 BOOLEAN DEFAULT FALSE,
    act_char_1_type_10 BOOLEAN DEFAULT FALSE,
    act_char_1_type_11 BOOLEAN DEFAULT FALSE,
    act_char_1_type_12 BOOLEAN DEFAULT FALSE,
    act_char_1_type_13 BOOLEAN DEFAULT FALSE,
    act_char_1_type_14 BOOLEAN DEFAULT FALSE,
    act_char_1_type_15 BOOLEAN DEFAULT FALSE,
    act_char_1_type_16 BOOLEAN DEFAULT FALSE,
    act_char_1_type_17 BOOLEAN DEFAULT FALSE,
    act_char_1_type_18 BOOLEAN DEFAULT FALSE,
    act_char_1_type_19 BOOLEAN DEFAULT FALSE,
    act_char_1_type_2 BOOLEAN DEFAULT FALSE,
    act_char_1_type_20 BOOLEAN DEFAULT FALSE,
    act_char_1_type_21 BOOLEAN DEFAULT FALSE,
    act_char_1_type_22 BOOLEAN DEFAULT FALSE,
    act_char_1_type_23 BOOLEAN DEFAULT FALSE,
    act_char_1_type_24 BOOLEAN DEFAULT FALSE,
    act_char_1_type_25 BOOLEAN DEFAULT FALSE,
    act_char_1_type_26 BOOLEAN DEFAULT FALSE,
    act_char_1_type_27 BOOLEAN DEFAULT FALSE,
    act_char_1_type_28 BOOLEAN DEFAULT FALSE,
    act_char_1_type_29 BOOLEAN DEFAULT FALSE,
    act_char_1_type_3 BOOLEAN DEFAULT FALSE,
    act_char_1_type_30 BOOLEAN DEFAULT FALSE,
    act_char_1_type_31 BOOLEAN DEFAULT FALSE,
    act_char_1_type_32 BOOLEAN DEFAULT FALSE,
    act_char_1_type_33 BOOLEAN DEFAULT FALSE,
    act_char_1_type_34 BOOLEAN DEFAULT FALSE,
    act_char_1_type_35 BOOLEAN DEFAULT FALSE,
    act_char_1_type_36 BOOLEAN DEFAULT FALSE,
    act_char_1_type_37 BOOLEAN DEFAULT FALSE,
    act_char_1_type_38 BOOLEAN DEFAULT FALSE,
    act_char_1_type_39 BOOLEAN DEFAULT FALSE,
    act_char_1_type_4 BOOLEAN DEFAULT FALSE,
    act_char_1_type_40 BOOLEAN DEFAULT FALSE,
    act_char_1_type_41 BOOLEAN DEFAULT FALSE,
    act_char_1_type_42 BOOLEAN DEFAULT FALSE,
    act_char_1_type_43 BOOLEAN DEFAULT FALSE,
    act_char_1_type_44 BOOLEAN DEFAULT FALSE,
    act_char_1_type_45 BOOLEAN DEFAULT FALSE,
    act_char_1_type_46 BOOLEAN DEFAULT FALSE,
    act_char_1_type_47 BOOLEAN DEFAULT FALSE,
    act_char_1_type_48 BOOLEAN DEFAULT FALSE,
    act_char_1_type_49 BOOLEAN DEFAULT FALSE,
    act_char_1_type_5 BOOLEAN DEFAULT FALSE,
    act_char_1_type_50 BOOLEAN DEFAULT FALSE,
    act_char_1_type_52 BOOLEAN DEFAULT FALSE,
    act_char_1_type_6 BOOLEAN DEFAULT FALSE,
    act_char_1_type_7 BOOLEAN DEFAULT FALSE,
    act_char_1_type_8 BOOLEAN DEFAULT FALSE,
    act_char_1_type_9 BOOLEAN DEFAULT FALSE,
    act_char_2_none BOOLEAN DEFAULT FALSE,
    act_char_2_type_1 BOOLEAN DEFAULT FALSE,
    act_char_2_type_10 BOOLEAN DEFAULT FALSE,
    act_char_2_type_11 BOOLEAN DEFAULT FALSE,
    act_char_2_type_12 BOOLEAN DEFAULT FALSE,
    act_char_2_type_13 BOOLEAN DEFAULT FALSE,
    act_char_2_type_14 BOOLEAN DEFAULT FALSE,
    act_char_2_type_15 BOOLEAN DEFAULT FALSE,
    act_char_2_type_16 BOOLEAN DEFAULT FALSE,
    act_char_2_type_17 BOOLEAN DEFAULT FALSE,
    act_char_2_type_18 BOOLEAN DEFAULT FALSE,
    act_char_2_type_19 BOOLEAN DEFAULT FALSE,
    act_char_2_type_2 BOOLEAN DEFAULT FALSE,
    act_char_2_type_20 BOOLEAN DEFAULT FALSE,
    act_char_2_type_21 BOOLEAN DEFAULT FALSE,
    act_char_2_type_22 BOOLEAN DEFAULT FALSE,
    act_char_2_type_23 BOOLEAN DEFAULT FALSE,
    act_char_2_type_24 BOOLEAN DEFAULT FALSE,
    act_char_2_type_25 BOOLEAN DEFAULT FALSE,
    act_char_2_type_26 BOOLEAN DEFAULT FALSE,
    act_char_2_type_27 BOOLEAN DEFAULT FALSE,
    act_char_2_type_28 BOOLEAN DEFAULT FALSE,
    act_char_2_type_29 BOOLEAN DEFAULT FALSE,
    act_char_2_type_3 BOOLEAN DEFAULT FALSE,
    act_char_2_type_30 BOOLEAN DEFAULT FALSE,
    act_char_2_type_31 BOOLEAN DEFAULT FALSE,
    act_char_2_type_32 BOOLEAN DEFAULT FALSE,
    act_char_2_type_4 BOOLEAN DEFAULT FALSE,
    act_char_2_type_5 BOOLEAN DEFAULT FALSE,
    act_char_2_type_6 BOOLEAN DEFAULT FALSE,
    act_char_2_type_7 BOOLEAN DEFAULT FALSE,
    act_char_2_type_8 BOOLEAN DEFAULT FALSE,
    act_char_2_type_9 BOOLEAN DEFAULT FALSE,
    act_char_3_none BOOLEAN DEFAULT FALSE,
    act_char_3_type_1 BOOLEAN DEFAULT FALSE,
    act_char_3_type_10 BOOLEAN DEFAULT FALSE,
    act_char_3_type_11 BOOLEAN DEFAULT FALSE,
    act_char_3_type_2 BOOLEAN DEFAULT FALSE,
    act_char_3_type_3 BOOLEAN DEFAULT FALSE,
    act_char_3_type_4 BOOLEAN DEFAULT FALSE,
    act_char_3_type_5 BOOLEAN DEFAULT FALSE,
    act_char_3_type_6 BOOLEAN DEFAULT FALSE,
    act_char_3_type_7 BOOLEAN DEFAULT FALSE,
    act_char_3_type_8 BOOLEAN DEFAULT FALSE,
    act_char_3_type_9 BOOLEAN DEFAULT FALSE,
    act_char_4_none BOOLEAN DEFAULT FALSE,
    act_char_4_type_1 BOOLEAN DEFAULT FALSE,
    act_char_4_type_2 BOOLEAN DEFAULT FALSE,
    act_char_4_type_3 BOOLEAN DEFAULT FALSE,
    act_char_4_type_4 BOOLEAN DEFAULT FALSE,
    act_char_4_type_5 BOOLEAN DEFAULT FALSE,
    act_char_4_type_6 BOOLEAN DEFAULT FALSE,
    act_char_4_type_7 BOOLEAN DEFAULT FALSE,
    act_char_5_none BOOLEAN DEFAULT FALSE,
    act_char_5_type_1 BOOLEAN DEFAULT FALSE,
    act_char_5_type_2 BOOLEAN DEFAULT FALSE,
    act_char_5_type_3 BOOLEAN DEFAULT FALSE,
    act_char_5_type_4 BOOLEAN DEFAULT FALSE,
    act_char_5_type_5 BOOLEAN DEFAULT FALSE,
    act_char_5_type_6 BOOLEAN DEFAULT FALSE,
    act_char_5_type_7 BOOLEAN DEFAULT FALSE,
    act_char_6_none BOOLEAN DEFAULT FALSE,
    act_char_6_type_1 BOOLEAN DEFAULT FALSE,
    act_char_6_type_2 BOOLEAN DEFAULT FALSE,
    act_char_6_type_3 BOOLEAN DEFAULT FALSE,
    act_char_6_type_4 BOOLEAN DEFAULT FALSE,
    act_char_6_type_5 BOOLEAN DEFAULT FALSE,
    act_char_7_none BOOLEAN DEFAULT FALSE,
    act_char_7_type_1 BOOLEAN DEFAULT FALSE,
    act_char_7_type_2 BOOLEAN DEFAULT FALSE,
    act_char_7_type_3 BOOLEAN DEFAULT FALSE,
    act_char_7_type_4 BOOLEAN DEFAULT FALSE,
    act_char_7_type_5 BOOLEAN DEFAULT FALSE,
    act_char_7_type_6 BOOLEAN DEFAULT FALSE,
    act_char_7_type_7 BOOLEAN DEFAULT FALSE,
    act_char_7_type_8 BOOLEAN DEFAULT FALSE,
    act_char_8_none BOOLEAN DEFAULT FALSE,
    act_char_8_type_1 BOOLEAN DEFAULT FALSE,
    act_char_8_type_10 BOOLEAN DEFAULT FALSE,
    act_char_8_type_11 BOOLEAN DEFAULT FALSE,
    act_char_8_type_12 BOOLEAN DEFAULT FALSE,
    act_char_8_type_13 BOOLEAN DEFAULT FALSE,
    act_char_8_type_14 BOOLEAN DEFAULT FALSE,
    act_char_8_type_15 BOOLEAN DEFAULT FALSE,
    act_char_8_type_16 BOOLEAN DEFAULT FALSE,
    act_char_8_type_17 BOOLEAN DEFAULT FALSE,
    act_char_8_type_18 BOOLEAN DEFAULT FALSE,
    act_char_8_type_2 BOOLEAN DEFAULT FALSE,
    act_char_8_type_3 BOOLEAN DEFAULT FALSE,
    act_char_8_type_4 BOOLEAN DEFAULT FALSE,
    act_char_8_type_5 BOOLEAN DEFAULT FALSE,
    act_char_8_type_6 BOOLEAN DEFAULT FALSE,
    act_char_8_type_7 BOOLEAN DEFAULT FALSE,
    act_char_8_type_8 BOOLEAN DEFAULT FALSE,
    act_char_8_type_9 BOOLEAN DEFAULT FALSE,
    act_char_9_none BOOLEAN DEFAULT FALSE,
    act_char_9_type_1 BOOLEAN DEFAULT FALSE,
    act_char_9_type_10 BOOLEAN DEFAULT FALSE,
    act_char_9_type_11 BOOLEAN DEFAULT FALSE,
    act_char_9_type_12 BOOLEAN DEFAULT FALSE,
    act_char_9_type_13 BOOLEAN DEFAULT FALSE,
    act_char_9_type_14 BOOLEAN DEFAULT FALSE,
    act_char_9_type_15 BOOLEAN DEFAULT FALSE,
    act_char_9_type_16 BOOLEAN DEFAULT FALSE,
    act_char_9_type_17 BOOLEAN DEFAULT FALSE,
    act_char_9_type_18 BOOLEAN DEFAULT FALSE,
    act_char_9_type_19 BOOLEAN DEFAULT FALSE,
    act_char_9_type_2 BOOLEAN DEFAULT FALSE,
    act_char_9_type_3 BOOLEAN DEFAULT FALSE,
    act_char_9_type_4 BOOLEAN DEFAULT FALSE,
    act_char_9_type_5 BOOLEAN DEFAULT FALSE,
    act_char_9_type_6 BOOLEAN DEFAULT FALSE,
    act_char_9_type_7 BOOLEAN DEFAULT FALSE,
    act_char_9_type_8 BOOLEAN DEFAULT FALSE,
    act_char_9_type_9 BOOLEAN DEFAULT FALSE,
    act_char_10 TEXT);