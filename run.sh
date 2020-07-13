
#python main.py --c ./config/train/baseline.yaml --mode baseline
#python main.py --c ./config/train/supervised_da.yaml --mode limit
#python main.py --c ./config/train/dagan.yaml --mode dagan
#python main.py --c ./config/train/pi_model.yaml --mode pimt
python main.py --c ./config/train/noisy_student.yaml --mode pimt
