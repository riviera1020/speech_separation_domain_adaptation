
#python ./data/vctk/preprocess.py \
#    --in_dir /home/riviera1020/Big/Corpus/vctk-mix/wav8k/min/ \
#    --out_dir ./data/vctk/id_list/

#python ./data/tgt2pkl.py \
#    --in_dir /home/riviera1020/Big/Corpus/VCTK-Corpus/train_force_aligned/pre_force/ \
#    --out_path ./data/vctk/phn.pkl

mkdir -p ./data/vctk/fa_id_list
python ./data/check_fa_exist.py \
    --dset vctk \
    --out_dir ./data/vctk/fa_id_list
