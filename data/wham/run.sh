
#mkdir -p ./data/wham/id_list

#python ./data/wham/preprocess.py \
#    --in_dir /home/riviera1020/Big/Corpus/wham-mix/wav8k/min/ \
#    --out_dir ./data/wham/id_list

mkdir -p ./data/wham/noise_id_list
python ./data/wham/fetch_scale.py \
    --noise_dir /home/riviera1020/Big/Corpus/wham_noise/ \
    --wsj0_dir /home/riviera1020/Big/Corpus/wsj0-mix/ \
    --out_dir ./data/wham/noise_id_list
