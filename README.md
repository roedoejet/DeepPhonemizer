<p align="center">
    <br>
    <img src="assets/header.png" width="800"/>
    <br>
</p>

<h2 align="center">
<p>A G2P library in PyTorch</p>
</h2>

![Build Status](https://github.com/as-ideas/DeepPhonemizer/workflows/pytest/badge.svg)
![codecov](https://codecov.io/gh/as-ideas/DeepPhonemizer/branch/main/graph/badge.svg)
![PyPI Version](https://img.shields.io/pypi/v/deep-phonemizer)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

DeepPhonemizer is a library for grapheme to phoneme conversion based on Transformer models. 
It is intended to be used in text-to-speech production systems with high accuracy and efficiency.
You can choose between a forward Transformer model (trained with CTC) and its autoregressive
counterpart. The former is faster and more stable while the latter is slightly more accurate.

The main advantages of this repo are:

* Easy-to-use API for training and inference.
* Multilingual: You can train a single model on several languages.
* Accuracy: Phoneme and word error rates are comparable to state-of-art. 
* Speed: The repo is highly optimized for fast inference by using dictionaries and batching.


Check out the [inference](https://colab.research.google.com/github/as-ideas/DeepPhonemizer/blob/main/deep_phonemizer/notebooks/Inference_Example.ipynb) and [training](https://colab.research.google.com/github/as-ideas/DeepPhonemizer/blob/main/deep_phonemizer/notebooks/Training_Example.ipynb) tutorials on Colab! 

Read the documentation at: https://as-ideas.github.io/DeepPhonemizer/


### Installation

```bash
pip install deep-phonemizer
```

### Quickstart

Download the pretrained model: [en_us_cmudict_ipa_forward](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa_forward.pt)

```bash
from deep_phonemizer.phonemizer import Phonemizer

phonemizer = Phonemizer.from_checkpoint('en_us_cmudict_ipa.pt')
phonemizer('Phonemizing an English text is imposimpable!', lang='en_us')

'foʊnɪmaɪzɪŋ æn ɪŋglɪʃ tɛkst ɪz ɪmpəzɪmpəbəl!'
```


### Training

You can easily train your own autoregressive or forward transformer model. 
All necessary parameters are set in a config.yaml, which you can find under:
```bash
deep_phonemizer/configs/forward_config.yaml
deep_phonemizer/configs/autoreg_config.yaml
```
for the forward and autoregressive transformer model, respectively.

Distributed training is supported. You can specify which GPUs to utilize by setting CUDA_VISIBLE_DEVICES env variable:
```bash
CUDA_VISIBLE_DEVICES=0,1 python run_training.py
```

Inside the training script prepare data in a tuple-format and use the preprocess and train API:

```python
from deep_phonemizer.preprocess import preprocess
from deep_phonemizer.train import train

train_data = [('en_us', 'young', 'jʌŋ'),
                ('de', 'benützten', 'bənʏt͡stn̩'),
                ('de', 'gewürz', 'ɡəvʏʁt͡s')] * 1000

val_data = [('en_us', 'young', 'jʌŋ'),
            ('de', 'benützten', 'bənʏt͡stn̩')] * 100

config_file = 'deep_phonemizer/configs/forward_config.yaml'

preprocess(config_file=config_file,
           train_data=train_data,
           val_data=val_data,
           deduplicate_train_data=False)

num_gpus = torch.cuda.device_count()

if num_gpus > 1:
    mp.spawn(train, nprocs=num_gpus, args=(num_gpus, config_file))
else:
    train(rank=0, num_gpus=num_gpus, config_file=config_file)
```
Model checkpoints will be stored in the checkpoints path that is provided by the config.yaml.

### Inference

Load the phonemizer from a checkpoint and run a prediction. By default, the phonemizer stores a 
dictionary of word-phoneme mappings that is applied first, and it uses the Transformer model
only to predict out-of-dictionary words.

```python
from deep_phonemizer.phonemizer import Phonemizer

phonemizer = Phonemizer.from_checkpoint('checkpoints/best_model.pt')
phonemes = phonemizer('Phonemizing an English text is imposimpable!', lang='en_us')
```

If you need more inference information, you can use following API:

```python
from deep_phonemizer.phonemizer import Phonemizer

result = phonemizer.phonemise_list(['Phonemizing an English text is imposimpable!'], lang='en_us')

for word, pred in result.predictions.items():
  print(f'{word} {pred.phonemes} {pred.confidence}')
```


### Pretrained Models

| Model | Language | Dataset | Repo Version
|---|---|---|---|
|[en_us_cmudict_ipa_forward](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa_forward.pt) | en_us | [cmudict-ipa](https://github.com/menelik3/cmudict-ipa) | 0.0.10 |
|[en_us_cmudict_forward](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_forward.pt) | en_us | [cmudict](https://github.com/microsoft/CNTK/tree/master/Examples/SequenceToSequence/CMUDict/Data) | 0.0.10 |
|[latin_ipa_forward](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/latin_ipa_forward.pt) | en_uk, en_us, de, fr, es | [wikipron](https://github.com/CUNY-CL/wikipron/tree/master/data/scrape/tsv) | 0.0.10 |

### Torchscript Export

You can easily export the underlying transformer models with TorchScript:
```python
import torch
from deep_phonemizer.phonemizer import Phonemizer

phonemizer = Phonemizer.from_checkpoint('checkpoints/best_model.pt')
model = phonemizer.predictor.model
phonemizer.predictor.model = torch.jit.script(model)
phonemizer('Running the torchscript model!')
```


### Maintainers
* Christian Schäfer, github: [cschaefer26](https://github.com/cschaefer26)


### References

[Transformer based Grapheme-to-Phoneme Conversion](https://arxiv.org/abs/2004.06338)

[GRAPHEME-TO-PHONEME CONVERSION USING
LONG SHORT-TERM MEMORY RECURRENT NEURAL NETWORKS](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43264.pdf)
