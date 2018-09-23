# [context2vec: Learning Generic Context Embedding with Bidirectional LSTM](http://www.aclweb.org/anthology/K16-1006), Melamud et al., CoNLL 2016

This is a PyTorch implementation of [Context2Vec](http://www.aclweb.org/anthology/K16-1006) that learns context vectors utilizing bi-directional LSTM.

## Requirements
### Framework
 - python (<= 3.6)
 - pytorch (<= 0.4.1)
 
### Packages
 - torchtext
 - nltk
 
## Quick Run
### Train
```
python -m src --train
```

which means running on cpu and learning context vectors from a small piece of penn tree bank (that is in the repository).
The learned model and the embedding file are stored at `models/model.param` and `models/embedding.vec` respectively.
(Note that you have to put flag `--train` if you want to train the model. Otherwise you might be on an inference mode.)

### Inference
```
python -m src
>> I am a [] .
```
(Note that you might not get a good result if you use the model that learns from a part of penn tree bank (i.e. `dataset/sample.txt`) because it does not contain enough data for learning context vectors. The reason why I put this sample in the repository is that you can easily check whether this program could actually work.)

## Running with GPU and other settings
### Train
Running on GPU_ID 0 using INPUT_FILE and output embedding file on OUTPUT_EMBEDDING_FILE and model parameters on MODEL_FILE.
(The other detailed settings are set by `config.toml`)

```
python -m src -g 0 -i INPUT_FILE -w OUTPUT_EMBEDDING_FILE -m MODEL_FILE --train
```

### Inference

```
python -m src -w WORD_EMBEDDING_FILE -m MODEL_FILE
```

## Performance
### Training Speed

There is approximatitely 3x speed up compared to the original implementation.

### MSR Sentence Completion
After setting your question/answer file in `config.toml`, run

```
python -m src --task mscc -w WORD_EMBEDDING_FILE -m MODEL_FILE
```

| - | Reported score | This implementation |
|:---:|:---:|:---:|
| TEST | 64.0 | 65.9 |
| ALL | 65.1 | 65.8 |

## Reference
 - The original implementation (written in Chainer) by the [author](https://researcher.watson.ibm.com/researcher/view.php?person=ibm-Oren.Melamud) can be found [here](https://github.com/orenmel/context2vec).

```
@InProceedings{K16-1006,
  author = 	"Melamud, Oren
		and Goldberger, Jacob
		and Dagan, Ido",
  title = 	"context2vec: Learning Generic Context Embedding with Bidirectional LSTM",
  booktitle = 	"Proceedings of The 20th SIGNLL Conference on Computational Natural Language Learning",
  year = 	"2016",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"51--61",
  location = 	"Berlin, Germany",
  doi = 	"10.18653/v1/K16-1006",
  url = 	"http://www.aclweb.org/anthology/K16-1006"
}
```
