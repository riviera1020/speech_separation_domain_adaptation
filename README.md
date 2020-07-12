# Toward Domain Generalization for Speech Separation
This is code for my thesis, which including pytorch implementation of Conv-TasNet
and several domain adaptation methods in my thesis.

## Data Preprosess

Check [here](data/make_mix).

## Requirements

* Python 3.7
* PyTorch 1.4.0
* `pip install -r requirements.txt`
* [Comet-ml](https://github.com/comet-ml/comet-examples) (Visualization)
* apex 0.1 (Deprecated)

## Basic Config

1. `cp config/path_example.yaml config/path.yaml`
2. Change path in config/path.yaml
3. Change information in .comet.config

## Usage

Training
```python
python main.py --c <config> --mode <mode>
```

Training Arugments

| Method | `<mode>` | Example Config |
| :------------:| :---------------: | :-----:|
| Baseline | baseline | config/train/baseline.yaml |
| Supervised Domain Adaptation | limit | |
| Domain Adversarial | dagan | |
| Pi-Model | pimt | |
| Noisy Teacher | pimt | |

---

Testing
```python
python main.py --c <config> --mode <mode> --test
```

## Config of training scripts

## Reference
