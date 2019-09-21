# MHAL configurations

We provide examples of configurations for our zero-shot sequence labeler MHAL-zero, our multi-task model MHAL-joint, and our single-task models MHAL-sent and MHAL-tok.

Edit the values in the config files (with .conf extension) as needed:

 * **load_pretrained_model** - Whether to load a pre-trained model or not (case in which a new one is trained)
 * **model_type** - Can be last, baseline, attention, or any name of a variant. 
 * **sentence_label** - Can be binary, majority or specified.
 * **default_label** - The most common (negative) label in the dataset. For example, the correct label in error detection or neutral label in sentiment detection. 
 * **token_labels_available** - Whether the token labels are available during training (e.g. for hedge detection they are not).
 * **plot_token_scores** - Whether to plot or not individual token scores as matplotlib figures.
 * **plot_predictions_html** - Whether to obtain visualizations for the attention weights and sentence predictions as html file.
 * **conll03_eval** - Set it to true when evaluating on CoNLL-2003 dataset, so the correct metric (span-sensitive) is picked out.
 * **to_write_filename** -  Path where to save the results.
 * **path_train** - Path to the training data, in CoNLL tab-separated format. Can contain multiple files, colon separated.
 * **path_dev** - Path to the development data, used for choosing the best epoch. Can contain multiple files, colon separated.
 * **path_test** - Path to the test file. Can contain multiple files, colon separated. 
 * **model_selector** - The development metric used as the stopping criterion. For example, it can be "f_score_micro_sent:high". You can specify more than one metric but the high/low should remain the last thing to be specified: e.g. "f_score_micro_tok:f_score_micro_sent:high" will optimize the dev micro f-score both on tokens and on sentences.
 * **model_selector_ratio** - The ratio between the various model selectors specified in "model_selector" separated by ":". Should be either one ratio (in which case, each model selector will have the same weight), or a total of ratio numbers equal to the number of model selectors specified by "model_selector". The ratios are implicitly normalized so that the sum of the ratios are linear combinations and add up to 1. e.g. 2:1 for the previous model selector will put a wight of 0.666 on the dev micro f-score of the tokens, and a weight of 0.333 on the dev micro f-scores of the sentences. 
 * **preload_vectors** - Path to the pre-trained word embeddings.
 * **word_embedding_size** - Size of the word embeddings used in the model.
 * **emb_initial_zero** - Whether word embeddings should be initialized with zeros. Otherwise, they are initialized randomly. If 'preload_vectors' is set, the initialization will be overwritten either way for words that have pretrained embeddings.
 * **train_embeddings** - Whether word embeddings are updated during training.
 * **char_embedding_size** - Size of the character embeddings.
 * **word_recurrent_size** - Size of the word-level LSTM hidden layers.
 * **char_recurrent_size** - Size of the char-level LSTM hidden layers.
 * **hidden_layer_size** - Final hidden layer size, right before word-level predictions.
 * **char_hidden_layer_size** - Char-level representation size, right before it gets combined with the word embeddings.
 * **lowercase** - Whether words should be lowercased.
 * **replace_digits** - Whether all digits should be replaced by zero.
 * **min_word_freq** - Minimal frequency of words to be included in the vocabulary. Others will be considered OOV.
 * **singletons_prob** - The probability with which words that occur only once are replaced with OOV during training.
 * **allowed_word_length** - Maximum allowed word length, clipping the rest.
 * **max_train_sent_length** - Discard sentences in the training set that are longer than this.
 * **vocab_include_devtest** - Whether the loaded vocabulary includes words also from the dev and test set. Since the word embeddings for these words are not updated during training, this is equivalent to preloading embeddings at test time as needed.
 * **vocab_only_embedded** - Whether to only include words in the vocabulary if they have pre-trained embeddings.
 * **initializer** - Method for random initialization. Choose between normal, glorot, and xavier.
 * **opt_strategy** - Optimization strategy used. Choose between adam, adadelta, and sgd.
 * **learning_rate** - Learning rate. 
 * **clip** - Gradient clip limit.
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
 * **lm_cost_max_vocab_size** - Maximum vocabulary size for the language modeling objective.
 * **lm_cost_hidden_layer_size** - Hidden layer size for LMCost.
 * **lm_cost_lstm_gamma** - LMCost weight
 * **lm_cost_joint_lstm_gamma** - Joint LMCost weight
 * **lm_cost_char_gamma** - Char-level LMCost weight
 * **lm_cost_joint_char_gamma** - Joint char-level LMCost weight
 * **char_integration_method** - Method for combining character-based representations with word embeddings.
 * **save** - Path for saving the model.
 * **garbage_collection** - Whether to force garbage collection.
 * **lstm_use_peepholes** - Whether LSTMs use the peephole architecture.
 * **whidden_layer_size** - Hidden layer size after the word-level LSTMs.
 * **attention_evidence_size** - Layer size for predicting attention weights.
 * **attention_activation** - Type of activation to apply for attention weights. 
 * **enable_label_smoothing** - Whether to enable label smoothing in the attention objective weigth or not.
 * **smoothing_epsilon** - The value of the epsilon in the label smoothign formula. Has no effect if label smoothing is not set.
 * **sentence_objective_weights_non_default** - How much weight to put on sentences of non-default type. 
 * **sentence_objective_weight** - How much weight to put on the sentence classification (main) loss.
 * **word_objective_weight**  - How much weight to put on the sequence labelling (main) loss.
 * **type1_attention_objective_weight** - How much weight to put on the first type of attention loss. This pushes the maximum head corresponding to the correct sentence label to be the highest, while tempering the other maximum heads.
 * **type2_attention_objective_weight** - How much weight to put on the first type of attention loss. This encourages the network to make the two predicted distributions (over the tokens and the sentences) similar.
 * **type3_attention_objective_weight** - How much weight to put on the first type of attention loss. This tells the network that at least one token has a label corresponding to the true sentence label.
 * **type4_attention_objective_weight** - How much weight to put on the first type of attention loss. This tells the network that a sentence that has a default label, should only contain tokens labeled as default.
 * **type5_attention_objective_weight** - How much weight to put on the first type of attention loss. This tells the network that every sentence has at least one default label.
 * **type6_attention_objective_weight** - How much weight to put on the first type of attention loss. This is a pairwise attention objective function.
 * **type7_attention_objective_weight** - How much weight to put on the first type of attention loss. This uses KL divergence to make the distribution over tokens similar to the distribution over sentences.
 * **regularize_queries** - How much weight to put on the query regularisation term.
 * **regularize_keys** - How much weight to put on the query regularisation term.
 * **regularize_values** - How much weight to put on the query regularisation term.
 * **regularize_sentence_repr** - How much weight to put on the query regularisation term.
 * **take_abs** - Whether to take the absolute value in the calculation of the cosine similarity. Has no effect if none of the regularisation terms is set.
 * **gap_objective_weight** - How much weight to put on the  gap-distance loss. This encourages a distance between the maximum default and maximum non-default heads.
 * **maximum_gap_threshold** - The difference between the default and non-default heads should be at least as big as this value. Has no effect if the gap_objective_weight was not set to a positive value.
 * **random_seed** - Random seed.
