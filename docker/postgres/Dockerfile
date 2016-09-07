FROM postgres
COPY tables.sql /docker-entrypoint-initdb.d/tables.sql
COPY act_test.csv /docker-entrypoint-init.d/act_test.csv
COPY act_train.csv /docker-entrypoint-init.d/act_train.csv
COPY people.csv /docker-entrypoint-init.d/people.csv

