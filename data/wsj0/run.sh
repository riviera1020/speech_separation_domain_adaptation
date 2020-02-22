
#python ./data/wsj0/preprocess.py \
#    --in_dir /home/riviera1020/Big/Corpus/wsj0-mix/ \
#    --out_dir ./data/wsj0/id_list/

python ./data/tgt2pkl.py \
    --in_dir /home/riviera1020/Big/Corpus/wsj0-wav/force_aligned/pre_force/ \
    --out_path ./data/wsj0/phn.pkl

mkdir -p ./data/wsj0/fa_id_list
python ./data/check_fa_exist.py \
    --dset wsj0 \
    --out_dir ./data/wsj0/fa_id_list
