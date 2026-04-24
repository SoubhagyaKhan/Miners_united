# python train_A.py \
#     --data_dir /home/trishanku/dm/Miners_united/A3/q2/public_datasets \
#     --model_dir ./best_models \
#     --kerberos aiy257584 \
#     --arch gcn

# python train_A.py \
#     --data_dir /home/trishanku/dm/Miners_united/A3/q2/public_datasets \
#     --model_dir ./best_models \
#     --kerberos aiy257584 \
#     --arch gat

# python train_B.py \
#     --data_dir /home/trishanku/dm/Miners_united/A3/q2/public_datasets \
#     --model_dir ./best_models \
#     --kerberos aiy257584 \

python train_C.py \
    --data_dir /home/trishanku/dm/Miners_united/A3/q2/public_datasets \
    --model_dir ./best_models \
    --kerberos aiy257584 \
    --decoder concat