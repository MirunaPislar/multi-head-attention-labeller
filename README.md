# Joint text classification on multiple levels with multiple labels

This repository contains the implementation for MHAL, a multi-head attention labeler that performs joint multi-class text classification on multiple levels (i.e., both at the sentence level and tokens level). As an instance of multi-task learning, MHAL uses hard parameter sharing to train two tasks under the same architecture. The token-level predictions are directly extracted from the attention evidence scores; the sentence representations are conditioned on the attention weights. Thus, the two tasks are intertwined and learned together.

Due to the design of its architecture, MHAL can perform reasonably well as a zero-shot sequence labeler -- without receiving any supervision signal on the sentence-level, it can perform sophisticated word-level classifications. For instance, given a sentence like "John Smith lives in Cambridge" whose label is "E" (meaning that it contains named entities) MHAL can label "John Smith" as a person and "Cambridge" as a location. MHAL is a robust and versatile model, being able to extract and share information between two levels of granularity. It has practical uses across many tasks. We tested it on named entity recognition, hedge detection, sentiment analysis, and grammatical error detection.

## How does it work?

There are two main components:
- Bi-LSTMs operating on characters and words, extracting representations for each token
- a multi-head attention mechanism, tying the token and sentence predictions together

We provide a schematic overview below, for one head *h* only.

![Architecture](plots/architecture.png =50x)

The main objectives are to train the model as a joint text classifier, but we also introduced some auxiliary objectives:
- char and/or word-based language modelling objectives
- an auxiliary loss called the *attention objective*, imposing that the attention weights reflect the ground truth of the sentence
- a query regularisation term, imposing the construction of a distinct subspace for each label in the tagset

## Requirements

* [Python](https://www.python.org/downloads/) (tested with 3.5.2)
* [Tensorflow](https://www.tensorflow.org/install) (tested with 1.13.1)
* [numpy](https://github.com/numpy/numpy)
* [pandas](https://pandas.pydata.org/pandas-docs/stable/install.html)
* [tqdm](https://github.com/tqdm/tqdm)
* some pre-trained word embeddings (I use [Glove](https://www.aclweb.org/anthology/D14-1162), which can be downloaded from [here](https://nlp.stanford.edu/projects/glove/))

## Intructions

For the main MHAL model, run:

```bash
    python3 experiment.py conf/example_of_config.conf
```

We also provide a wide range of variants. Most of them are just experimental work and have not been tested thoroughly. However, if you'd like to try any of the variants, I provide the config file. Make sure you comment this line ```from model import Model``` and uncomment this line ```from variants import Model``` in *experiment.py*, and run:

```bash
    python3 experiment.py conf/example_of_config_for_variants.conf
```

## Data format

The training and test data is expected in standard CoNLL-type. There is one word per line, with an arbitrary number of tab-separated values. The first value must be the word, the last one is its label. Each sentence is separated by an extra empty line. If the sentence labels are also specified, then they precede all the constituent words and have to start with *sent_label* followed by an equal number of columns as the tokens, the last one being the sentence label. Here is an example of sentiment detection:

    I       O
    loved    P
    his     O
    performance    O
    sent_label       P

However, if the sentence label is not explicitly specified, an implicit binary labelling can be performed. For instance, the example from an error detection task given below will be assigned a *positive* (i.e. it is ungrammatical) sentence label:

    I    c
    was    c
    really    c
    disappointing    FORM
    in    FUNCTION
    many    c
    points    CONTENT
    .    c

## Acknowledgements and references

I would like to thank my supervisor, Dr Marek Rei, for inspiring this project and guiding me throughout. The code in this repository has largely been structured based on his other two projects: [sequence-labeler](https://github.com/marekrei/sequence-labeler) and [mltagger](https://github.com/marekrei/mltagger).

[**Jointly Learning to Label Sentences and Tokens**](https://arxiv.org/pdf/1811.05949.pdf) by Marek Rei and Anders Søgaard (2019)

[**Zero-Shot Sequence Labeling: Transferring Knowledge from Sentences to Tokens**](https://www.aclweb.org/anthology/N18-1027)  by Marek Rei and Anders Søgaard (2018)

[**Semi-supervised Multitask Learning for Sequence Labeling**](https://arxiv.org/abs/1704.07156) by Marek Rei (2017)

[**Attention Is All You Need**](https://arxiv.org/pdf/1706.03762.pdf) by Ashish Vaswani et al. (2017)

[**Neural Architectures for Named Entity Recognition**](https://www.aclweb.org/anthology/N16-1030) by Lample et al. (2016)

## License
Everything is licensed under the MIT license.
