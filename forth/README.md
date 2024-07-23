
/usr/local/bin/python3 -W ignore forth/TrainingLocalDQL.py


nohup python3 -W ignore forth/TrainingLocalDQL.py > out1.file 2>&1 &

nohup python3 -W ignore forth/TrainingLocalDQL3_26_best_copy.py > out1.file 2>&1 &


nohup sh forth/train_auto.sh  > out_sh0513.file 2>&1 &