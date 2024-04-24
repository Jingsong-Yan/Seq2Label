model='bart' # ['bart', 't5']
gpu_id=0
dataset="rcv1" # [wos, rcv1, bgc]
lr=2e-5 # bart: 2e-5, t5: 2e-4
mode='random_each_epoch' # [random, bfs, dfs, random_each_epoch]
data_path='/.../dataset' # the root directory of the three datasets
log_path='./log/'${model}'/'${dataset}
reversed=0 # 是否把标签顺序倒过来
save_model_path='./outputs/'${model}'/'${dataset}'/'
train=1
test=1
# negative_sample = 1, random_negative_sample = 0 : Hierarchy-Aware Negative Sampling
# negative_sample = 0, random_negative_sample = 1 : 随机采样非目标标签
# 其余情况不采样，loss 为 CrossEntropyLoss
negative_sample=1
random_negative_sample=0

epochs=100
beam=1
train_batch_size=16
valid_batch_size=128
times=0
name=${dataset}'_'${mode}'_'${times}
# You need to change the path to your own environment.
CUDA_VISIBLE_DEVICES=${gpu_id} python -u train.py \
    --model ${model} \
    --gpu_id ${gpu_id} \
    --dataset ${dataset} \
    --mode ${mode} \
    --data_path ${data_path} \
    --log_path ${log_path} \
    --name ${name} \
    --reversed ${reversed} \
    --lr ${lr} \
    --save_model_path ${save_model_path} \
    --train ${train} \
    --test ${test} \
    --negative_sample ${negative_sample} \
    --random_negative_sample ${random_negative_sample} \
    --epochs ${epochs} \
    --beam ${beam} \
    --train_batch_size ${train_batch_size} \
    --valid_batch_size ${valid_batch_size} \
