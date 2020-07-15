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
* [TSNE-CUDA](https://github.com/CannyLab/tsne-cuda)
* apex 0.1 (Deprecated)

## Basic Config

1. `cp config/path_example.yaml config/path.yaml`
2. Change path in config/path.yaml
    - `wsj_root`: wsj0-2mix directory
    - `vctk_root`: vctk-2mix directory
    - `wsj0-vctk_root`: wsj0-vctk directory
    - Noise of Wham is moved to `wsj_root` based on [Data Process](https://github.com/riviera1020/speech_separation_domain_adaptation/tree/master/data/make_mix#wham)
    - All directory must contain 'tr/', 'cv/' and 'tt/' subfolder.

## Comet Introduction
Comet is a visualization tool. You can ignore this part if you don't need
visualization.


1. Create Comet account. [Link](https://www.comet.ml/docs/quick-start/#quick-start-fhttps://www.comet.ml/docs/quick-start/#quick-start-for-pythonor-python) ( Student Plan Suggested )
2. Create Comet Project.
3. Change information in .comet.config 
    - `api_key`: API Key of your account
    - `workspace`: Account name
    - `project_name`: Your project name

## Usage

Training
```python
python main.py --c <config> --mode <mode>
```

Training Arugments

| Method | `<mode>` | `<config>` |
| :------------ | :--------------- | :-----|
| Baseline | baseline | config/train/baseline.yaml |
| Supervised Domain Adaptation | limit | config/train/supervised_da.yaml |
| Domain Adversarial | dagan | config/train/dagan.yaml |
| Pi-Model | pimt | config/train/pi_model.yaml |
| Noisy Student | pimt | config/train/noisy_student.yaml |

---

Testing
```python
python main.py --c <config> --mode <mode> --test
```
Testing Arugments

| Method | `<mode>` | `<config>` |
| :------------ | :--------------- | :-----|
| Baseline | baseline | config/test/baseline.yaml |
| Supervised Domain Adaptation | baseline | config/test/baseline.yaml |
| Domain Adversarial | dagan | config/test/dagan.yaml |
| T-SNE (For dagan) | dacluster | config/test/tsne.yaml |
| Pi-Model | baseline | config/test/baseline.yaml |
| Noisy Student | baseline | config/test/baseline.yaml |

## Hyperparameters of training/testing config

### Training Config
Basic training config is consist of `data`, `model`, `optim` and `solver`.
`data`, `model` and `optim` provide hyperparameter for each part. `solver`
provide setting for every algorithm. Please check `config/train` for detailed
information.

Config for domain adversarial method is a little bit different. `model` is
split into `gen` and `dis`, which stand for generator and discrimator module in
GAN. All Conv-TasNet hyparameters is setting in `gen`. Also, `optim` is split
into `g_optim` and `d_optim` for optimizer setting of Conv-TasNet and
discriminator.

### Testing Config
Basic testing config is consist of `data` and `solver`. Detailed information is
in `config/test`.

## Reference

Links for code reference

* [Conv-TasNet by Kaituo](https://github.com/kaituoxu/Conv-TasNet)
* [Comet Intro by sunprinceS](https://sunprinces.github.io/learning/2020/01/comet-ml-%E4%BD%A0%E5%BF%85%E9%A0%88%E7%9F%A5%E9%81%93%E7%9A%84-ml-%E5%AF%A6%E9%A9%97%E7%AE%A1%E7%90%86%E7%A5%9E%E5%99%A8/)
* [Data Process of WSJ0-2mix](https://github.com/r06944010/Speech-Separation-TF2)
