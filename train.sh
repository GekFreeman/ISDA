d1=`date +%Y_%m_%d_%H_%M_%S`
# /root/anaconda3/envs/isda/bin/python train_da.py > logs/${d1}.txt 2>&1 
python train_da.py > logs/${d1}.txt 2>&1 &