split -l 550000 act_train.csv
head -n 1 xaa > act_train_2.csv 
mv xaa act_train_1.csv
cp act_train_2.csv act_train_3.csv
cp act_train_2.csv act_train_4.csv
cat xab >> act_train_2.csv
cat xac >> act_train_3.csv
cat xad >> act_train_4.csv
rm xab xac xad