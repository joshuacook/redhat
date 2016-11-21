FROM postgres
COPY tables.sql /docker-entrypoint-initdb.d/tables.sql
COPY new_act_test.csv /docker-entrypoint-init.d/act_test.csv
COPY new_act_train_1.csv /docker-entrypoint-init.d/act_train_1.csv
COPY new_act_train_2.csv /docker-entrypoint-init.d/act_train_2.csv
COPY new_act_train_3.csv /docker-entrypoint-init.d/act_train_3.csv
COPY new_people.csv /docker-entrypoint-init.d/people.csv

