# multi-label-multi-level-sequence-labeler
Multi-label and multi-level sequence labeler.

Configuration
-------------------------

Edit the values in config.conf as needed:

* **debug_mode** - set if need to debug or not
* **shrink_input** - in case the transformer architecture is used, choose to shrink it or not
* **model_type** = transformer, multi-head or previous
* **sentence_label** = binary, majority or specified
* **path_train** - Path to the training data, in CoNLL tab-separated format. One word per line, first column is the word, last column is the label. Empty lines between sentences.
* **path_dev** - Path to the development data, used for choosing the best epoch.
* **path_test** - Path to the test file. Can contain multiple files, colon separated.
* **default_label** - The most common (negative) label in the dataset. For example, the correct label in error detection or neutral label in sentiment detection.
* **model_selector** - What is measured on the dev set for model selection. For example, "dev_sent_f:high" means we're looking for the highest sentence-level F score on the development set. Can also have more than one model selectors but the high/low should remain the last thing to be specified: e.g. "dev_f-score_macro_sent:dev_f-score_macro_tok:high" will optimize both the macro f-score at the token and at the sentence levels.
* **model_selector_ratio** - The ratio between the various model selectors specified in "model_selector" separated by ":". Should specify either one ratio (in which case, each model selector will have the same weight), or a total of ratio numbers equal to the number of model selectors specified by "model_selector". The ratios are implicitly normalized so that the sum of the ratios are linear combinations and add up to 1. e.g. 2:1 for the previous model selector will put a wight of 0.666 on the dev macro f-score of the sentences, and a weight of 0.333 on the dev macro f-scores of the tokens. 
* **preload_vectors** - Path to the pre-trained word embeddings, in word2vec plain text format. If your embeddings are in binary, you can use [convertvec](https://github.com/marekrei/convertvec) to convert them to plain text.
* **word_embedding_size** - Size of the word embeddings used in the model.
* **emb_initial_zero** - Whether word embeddings should be initialized with zeros. Otherwise, they are initialized randomly. If 'preload_vectors' is set, the initialization will be overwritten either way for words that have pretrained embeddings.
* **train_embeddings** - Whether word embeddings are updated during training.
* **char_embedding_size** - Size of the character embeddings.
* **word_recurrent_size** - Size of the word-level LSTM hidden layers.
* **char_recurrent_size** - Size of the char-level LSTM hidden layers.
* **hidden_layer_size** - Final hidden layer size, right before word-level predictions.
* **char_hidden_layer_size** - Char-level representation size, right before it gets combined with the word embeddings.
* **lowercase** - Whether words should be lowercased.
* **replace_digits** - Whether all digits should be replaced by zeros.
* **min_word_freq** - Minimal frequency of words to be included in the vocabulary. Others will be considered OOV.
* **singletons_prob** - The probability with which words that occur only once are replaced with OOV during training.
* **allowed_word_length** - Maximum allowed word length, clipping the rest. Can be necessary if the text contains unreasonably long tokens, eg URLs.
* **max_train_sent_length** - Discard sentences in the training set that are longer than this.
* **vocab_include_devtest** - Whether the loaded vocabulary includes words also from the dev and test set. Since the word embeddings for these words are not updated during training, this is equivalent to preloading embeddings at test time as needed. This seems common practice for many sequence labeling toolkits, so I've included it as well. 
* **vocab_only_embedded** - Whether to only include words in the vocabulary if they have pre-trained embeddings.
* **initializer** - Method for random initialization
* **opt_strategy** - Optimization methods, e.g. adam, adadelta, sgd.
* **learning_rate** - Learning rate
* **clip** - Gradient clip limit
* **batch_equal_size** - Whether to construct batches from sentences of equal length.
* **max_batch_size** - Maximum batch size.
* **epochs** - Maximum number of epochs to run.
* **stop_if_no_improvement_for_epochs** - Stop if there has been no improvement for this many epochs.
* **learning_rate_decay** - Learning rate decay when performance hasn't improved.
* **dropout_input** - Apply dropout to word representations.
* **dropout_word_lstm** - Apply dropout after the LSTMs.
* **dropout_attention** - Apply dropout after computing the attention weights (i.e. after softmax) in the transformer architecture.
* **tf_per_process_gpu_memory_fraction** - Set 'tf_per_process_gpu_memory_fraction' for TensorFlow.
* **tf_allow_growth** - Set 'allow_growth' for TensorFlow
* **lmcost_max_vocab_size** - Maximum vocabulary size for the language modeling objective.
* **lmcost_hidden_layer_size** - Hidden layer size for LMCost.
* **lmcost_lstm_gamma** - LMCost weight
* **lmcost_joint_lstm_gamma** - Joint LMCost weight
* **lmcost_char_gamma** - Char-level LMCost weight
* **lmcost_joint_char_gamma** - Joint char-level LMCost weight
* **char_integration_method** - Method for combining character-based representations with word embeddings.
* **save** - Path for saving the model.
* **garbage_collection** - Whether to force garbage collection.
* **lstm_use_peepholes** - Whether LSTMs use the peephole architecture.
* **whidden_layer_size** - Hidden layer size after the word-level LSTMs.
* **attention_evidence_size** - Layer size for predicting attention weights.
* **attention_activation** - Type of activation to apply for attention weights.
* **attention_objective_weight** - The weight for pushing the attention weights to a binary classification range.
* **sentence_objective_weight** - Sentence-level objective weight.
* **sentence_objective_persistent** - Whether the sentence-level objective should always be given to the network.
* **word_objective_weight** - Word-level classification objective weight.
* **sentence_composition** - The method for sentence composition.
* **random_seed** - Random seed.
* **attention_loss_between_all** - boolean value; if the attention should be computed between all differences 
(i.e. should have maximum token close to one, all other maximums close to 0.)
* **gap_objective_weight** - objective weight encouraging a certain gap between the maximum default and non-default heads
* **maximum_gap_threshold** - the difference between the default and non-default heads should be at least as big as this value

