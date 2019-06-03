from modules import *
import collections
import numpy
import pickle
import re
import tensorflow as tf


class Model(object):
    """
    Implements several variants of the multi-head attention labeller (MHAL).
    These were mainly experimental, so don't take them as granted.
    The performances reported are obtained with the main model, "model.py".
    """

    def __init__(self, config, label2id_sent, label2id_tok):
        self.config = config
        self.label2id_sent = label2id_sent
        self.label2id_tok = label2id_tok

        self.UNK = "<unk>"
        self.CUNK = "<cunk>"
        self.word2id = None
        self.char2id = None
        self.singletons = None
        self.num_heads = None

        self.word_ids = None
        self.char_ids = None
        self.sentence_lengths = None
        self.word_lengths = None

        self.sentence_labels = None
        self.word_labels = None

        self.word_embeddings = None
        self.char_embeddings = None

        self.word_objective_weights = None
        self.learning_rate = None
        self.loss = None
        self.initializer = None
        self.is_training = None
        self.session = None
        self.saver = None
        self.train_op = None

        self.token_scores = None
        self.sentence_scores = None
        self.token_predictions = None
        self.sentence_predictions = None
        self.token_probabilities = None
        self.sentence_probabilities = None
        self.attention_weights = None

    def build_vocabs(self, data_train, data_dev, data_test, embedding_path=None):
        """
        Builds the vocabulary based on the the data and embeddings info.
        """
        data_source = list(data_train)
        if self.config["vocab_include_devtest"]:
            if data_dev is not None:
                data_source += data_dev
            if data_test is not None:
                data_source += data_test

        char_counter = collections.Counter()
        word_counter = collections.Counter()
        for sentence in data_source:
            for token in sentence.tokens:
                char_counter.update(token.value)
                w = token.value
                if self.config["lowercase"]:
                    w = w.lower()
                if self.config["replace_digits"]:
                    w = re.sub(r'\d', '0', w)
                word_counter[w] += 1

        self.char2id = collections.OrderedDict([(self.CUNK, 0)])
        for char, count in char_counter.most_common():
            if char not in self.char2id:
                self.char2id[char] = len(self.char2id)

        self.word2id = collections.OrderedDict([(self.UNK, 0)])
        for word, count in word_counter.most_common():
            if self.config["min_word_freq"] <= 0 or count >= self.config["min_word_freq"]:
                if word not in self.word2id:
                    self.word2id[word] = len(self.word2id)

        self.singletons = set([word for word in word_counter if word_counter[word] == 1])

        if embedding_path and self.config["vocab_only_embedded"]:
            embedding_vocab = {self.UNK}
            with open(embedding_path) as f:
                for line in f:
                    line_parts = line.strip().split()
                    if len(line_parts) <= 2:
                        continue
                    w = line_parts[0]
                    if self.config["lowercase"]:
                        w = w.lower()
                    if self.config["replace_digits"]:
                        w = re.sub(r'\d', '0', w)
                    embedding_vocab.add(w)
            word2id_revised = collections.OrderedDict()
            for word in self.word2id:
                if word in embedding_vocab and word not in word2id_revised:
                    word2id_revised[word] = len(word2id_revised)
            self.word2id = word2id_revised

        print("Total number of words: " + str(len(self.word2id)))
        print("Total number of chars: " + str(len(self.char2id)))
        print("Total number of singletons: " + str(len(self.singletons)))

    def construct_network(self):
        """
        Constructs a certain variant of the multi-head attention labeller (MHAL).
        """
        self.word_ids = tf.placeholder(tf.int32, [None, None], name="word_ids")
        self.char_ids = tf.placeholder(tf.int32, [None, None, None], name="char_ids")
        self.sentence_lengths = tf.placeholder(tf.int32, [None], name="sentence_lengths")
        self.word_lengths = tf.placeholder(tf.int32, [None, None], name="word_lengths")
        self.sentence_labels = tf.placeholder(tf.float32, [None], name="sentence_labels")
        self.word_labels = tf.placeholder(tf.float32, [None, None], name="word_labels")

        self.word_objective_weights = tf.placeholder(
            tf.float32, [None, None], name="word_objective_weights")

        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        self.is_training = tf.placeholder(tf.int32, name="is_training")
        self.loss = 0.0

        if self.config["initializer"] == "normal":
            self.initializer = tf.random_normal_initializer(stddev=0.1)
        elif self.config["initializer"] == "glorot":
            self.initializer = tf.glorot_uniform_initializer()
        elif self.config["initializer"] == "xavier":
            self.initializer = tf.glorot_normal_initializer()

        zeros_initializer = tf.zeros_initializer()

        self.word_embeddings = tf.get_variable(
            name="word_embeddings",
            shape=[len(self.word2id), self.config["word_embedding_size"]],
            initializer=(zeros_initializer if self.config["emb_initial_zero"] else self.initializer),
            trainable=(True if self.config["train_embeddings"] else False))
        word_input_tensor = tf.nn.embedding_lookup(self.word_embeddings, self.word_ids)

        if self.config["char_embedding_size"] > 0 and self.config["char_recurrent_size"] > 0:
            with tf.variable_scope("chars"), tf.control_dependencies(
                    [tf.assert_equal(tf.shape(self.char_ids)[2],
                                     tf.reduce_max(self.word_lengths),
                                     message="Char dimensions don't match")]):
                self.char_embeddings = tf.get_variable(
                    name="char_embeddings",
                    shape=[len(self.char2id), self.config["char_embedding_size"]],
                    initializer=self.initializer,
                    trainable=True)
                char_input_tensor = tf.nn.embedding_lookup(self.char_embeddings, self.char_ids)

                char_input_tensor_shape = tf.shape(char_input_tensor)
                char_input_tensor = tf.reshape(
                    char_input_tensor,
                    shape=[char_input_tensor_shape[0]
                           * char_input_tensor_shape[1],
                           char_input_tensor_shape[2],
                           self.config["char_embedding_size"]])
                _word_lengths = tf.reshape(
                    self.word_lengths, shape=[char_input_tensor_shape[0]
                                              * char_input_tensor_shape[1]])

                char_lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(
                    self.config["char_recurrent_size"],
                    use_peepholes=self.config["lstm_use_peepholes"],
                    state_is_tuple=True,
                    initializer=self.initializer,
                    reuse=False)
                char_lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(
                    self.config["char_recurrent_size"],
                    use_peepholes=self.config["lstm_use_peepholes"],
                    state_is_tuple=True,
                    initializer=self.initializer,
                    reuse=False)

                # Concatenate the final forward and the backward character contexts
                # to obtain a compact character representation for each word.
                _, ((_, char_output_fw), (_, char_output_bw)) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=char_lstm_cell_fw, cell_bw=char_lstm_cell_bw, inputs=char_input_tensor,
                    sequence_length=_word_lengths, dtype=tf.float32, time_major=False)

                char_output_tensor = tf.concat([char_output_fw, char_output_bw], axis=-1)
                char_output_tensor = tf.reshape(
                    char_output_tensor,
                    shape=[char_input_tensor_shape[0], char_input_tensor_shape[1],
                           2 * self.config["char_recurrent_size"]])

                # Include a char-based language modelling loss, LMc.
                if self.config["lm_cost_char_gamma"] > 0.0:
                    self.loss += self.config["lm_cost_char_gamma"] * \
                                 self.construct_lm_cost(
                                     input_tensor_fw=char_output_tensor,
                                     input_tensor_bw=char_output_tensor,
                                     sentence_lengths=self.sentence_lengths,
                                     target_ids=self.word_ids,
                                     lm_cost_type="separate",
                                     name="lm_cost_char_separate")

                if self.config["lm_cost_joint_char_gamma"] > 0.0:
                    self.loss += self.config["lm_cost_joint_char_gamma"] * \
                                 self.construct_lm_cost(
                                     input_tensor_fw=char_output_tensor,
                                     input_tensor_bw=char_output_tensor,
                                     sentence_lengths=self.sentence_lengths,
                                     target_ids=self.word_ids,
                                     lm_cost_type="joint",
                                     name="lm_cost_char_joint")

                if self.config["char_hidden_layer_size"] > 0:
                    char_output_tensor = tf.layers.dense(
                        inputs=char_output_tensor, units=self.config["char_hidden_layer_size"],
                        activation=tf.tanh, kernel_initializer=self.initializer)

                if self.config["char_integration_method"] == "concat":
                    word_input_tensor = tf.concat([word_input_tensor, char_output_tensor], axis=-1)
                elif self.config["char_integration_method"] == "none":
                    word_input_tensor = word_input_tensor
                else:
                    raise ValueError("Unknown char integration method")

        if self.config["dropout_input"] > 0.0:
            dropout_input = (self.config["dropout_input"] * tf.cast(self.is_training, tf.float32)
                             + (1.0 - tf.cast(self.is_training, tf.float32)))
            word_input_tensor = tf.nn.dropout(
                word_input_tensor, dropout_input, name="dropout_word")

        word_lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(
            self.config["word_recurrent_size"],
            use_peepholes=self.config["lstm_use_peepholes"],
            state_is_tuple=True,
            initializer=self.initializer,
            reuse=False)
        word_lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(
            self.config["word_recurrent_size"],
            use_peepholes=self.config["lstm_use_peepholes"],
            state_is_tuple=True,
            initializer=self.initializer,
            reuse=False)

        with tf.control_dependencies(
                [tf.assert_equal(
                    tf.shape(self.word_ids)[1],
                    tf.reduce_max(self.sentence_lengths),
                    message="Sentence dimensions don't match")]):
            (lstm_outputs_fw, lstm_outputs_bw), ((_, lstm_output_fw), (_, lstm_output_bw)) = \
                tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=word_lstm_cell_fw, cell_bw=word_lstm_cell_bw, inputs=word_input_tensor,
                    sequence_length=self.sentence_lengths, dtype=tf.float32, time_major=False)

        lstm_output_states = tf.concat([lstm_output_fw, lstm_output_bw], -1)

        if self.config["dropout_word_lstm"] > 0.0:
            dropout_word_lstm = (self.config["dropout_word_lstm"] * tf.cast(self.is_training, tf.float32)
                                 + (1.0 - tf.cast(self.is_training, tf.float32)))
            lstm_outputs_fw = tf.nn.dropout(
                lstm_outputs_fw, dropout_word_lstm,
                noise_shape=tf.convert_to_tensor(
                    [tf.shape(self.word_ids)[0], 1, self.config["word_recurrent_size"]], dtype=tf.int32))
            lstm_outputs_bw = tf.nn.dropout(
                lstm_outputs_bw, dropout_word_lstm,
                noise_shape=tf.convert_to_tensor(
                    [tf.shape(self.word_ids)[0], 1, self.config["word_recurrent_size"]], dtype=tf.int32))
            lstm_output_states = tf.nn.dropout(lstm_output_states, dropout_word_lstm)

        # The forward and backward states are concatenated at every token position.
        lstm_outputs_states = tf.concat([lstm_outputs_fw, lstm_outputs_bw], -1)  # [B, M, 2 * emb_size]

        if self.config["whidden_layer_size"] > 0:
            lstm_output_units = self.config["whidden_layer_size"]
            num_heads = len(self.label2id_tok)

            # Make the number of units a multiple of num_heads.
            if lstm_output_units % num_heads != 0:
                lstm_output_units = ceil(lstm_output_units / num_heads) * num_heads

            lstm_outputs = tf.layers.dense(
                inputs=lstm_outputs_states, units=lstm_output_units,
                activation=tf.tanh, kernel_initializer=self.initializer)  # [B, M, lstm_output_units]
        else:
            lstm_outputs = lstm_outputs_states

        if self.config["model_type"] == "single_head_attention_binary_labels":
            if not (len(self.label2id_tok) == 2 and len(self.label2id_sent) == 2):
                raise ValueError(
                    "The model_type you selected (%s) is only available for "
                    "binary labels! Currently, the no. sentence_labels = %d and "
                    "the no. token_labels = %d. Consider changing the model type."
                    % (self.config["model_type"],
                       len(self.label2id_sent), len(self.label2id_tok)))

            self.sentence_scores, self.sentence_predictions, \
                self.token_scores, self.token_predictions = \
                single_head_attention_binary_labels(
                    inputs=lstm_outputs,
                    initializer=self.initializer,
                    attention_size=self.config["attention_evidence_size"],
                    sentence_lengths=self.sentence_lengths,
                    hidden_units=self.config["hidden_layer_size"])

            # Include a token-level loss (for sequence labelling).
            word_objective_loss = tf.square(self.token_scores - self.word_labels)
            word_objective_loss = tf.where(
                tf.sequence_mask(self.sentence_lengths),
                word_objective_loss, tf.zeros_like(word_objective_loss))
            self.loss += self.config["word_objective_weight"] * tf.reduce_sum(
                self.word_objective_weights * word_objective_loss)

            # Include a sentence-level loss (for sentence classification).
            sentence_objective_loss = tf.square(self.sentence_scores - self.sentence_labels)
            self.loss += self.config["sentence_objective_weight"] * tf.reduce_sum(sentence_objective_loss)

            # Include an attention-level loss for wiring the two hierarchical levels.
            if self.config["attention_objective_weight"] > 0.0:
                self.loss += self.config["attention_objective_weight"] * (
                    tf.reduce_sum(
                        tf.square(
                            tf.reduce_max(
                                tf.where(
                                    tf.sequence_mask(self.sentence_lengths),
                                    self.token_scores,
                                    tf.zeros_like(self.token_scores) - 1e6),
                                axis=-1) - self.sentence_labels))
                    +
                    tf.reduce_sum(
                        tf.square(
                            tf.reduce_min(
                                tf.where(
                                    tf.sequence_mask(self.sentence_lengths),
                                    self.token_scores,
                                    tf.zeros_like(self.token_scores) + 1e6),
                                axis=-1) - 0.0)))
        else:
            scoring_activation = None
            if "scoring_activation" in self.config and self.config["scoring_activation"]:
                if self.config["scoring_activation"] == "tanh":
                    scoring_activation = tf.tanh
                elif self.config["scoring_activation"] == "sigmoid":
                    scoring_activation = tf.sigmoid
                elif self.config["scoring_activation"] == "relu":
                    scoring_activation = tf.nn.relu
                elif self.config["scoring_activation"] == "softmax":
                    scoring_activation = tf.nn.softmax

            if "baseline_lstm_last_contexts" in self.config["model_type"]:
                self.sentence_scores, self.sentence_predictions, \
                    self.token_scores, self.token_predictions, \
                    self.token_probabilities, self.sentence_probabilities, \
                    self.attention_weights = baseline_lstm_last_contexts(
                        last_token_contexts=lstm_outputs_states,
                        last_context=lstm_output_states,
                        initializer=self.initializer,
                        scoring_activation=scoring_activation,
                        sentence_lengths=self.sentence_lengths,
                        hidden_units=self.config["hidden_layer_size"],
                        num_sentence_labels=len(self.label2id_sent),
                        num_token_labels=len(self.label2id_tok))
            elif self.config["model_type"] == "single_head_attention_multiple_labels":
                self.sentence_scores, self.sentence_predictions, \
                    self.token_scores, self.token_predictions, \
                    self.token_probabilities, self.sentence_probabilities, \
                    self.attention_weights = single_head_attention_multiple_labels(
                        inputs=lstm_outputs,
                        initializer=self.initializer,
                        attention_activation=self.config["attention_activation"],
                        attention_size=self.config["attention_evidence_size"],
                        sentence_lengths=self.sentence_lengths,
                        hidden_units=self.config["hidden_layer_size"],
                        num_sentence_labels=len(self.label2id_sent),
                        num_token_labels=len(self.label2id_tok))
            elif self.config["model_type"] == "multi_head_attention_with_scores_from_shared_heads":
                self.sentence_scores, self.sentence_predictions, \
                    self.token_scores, self.token_predictions, \
                    self.token_probabilities, self.sentence_probabilities, \
                    self.attention_weights = multi_head_attention_with_scores_from_shared_heads(
                        inputs=lstm_outputs,
                        initializer=self.initializer,
                        attention_activation=self.config["attention_activation"],
                        hidden_units=self.config["hidden_layer_size"],
                        num_sentence_labels=len(self.label2id_sent),
                        num_heads=len(self.label2id_tok),
                        is_training=self.is_training,
                        dropout=self.config["dropout_attention"],
                        sentence_lengths=self.sentence_lengths,
                        use_residual_connection=self.config["residual_connection"],
                        token_scoring_method=self.config["token_scoring_method"])
            elif self.config["model_type"] == "multi_head_attention_with_scores_from_separate_heads":
                self.sentence_scores, self.sentence_predictions, \
                    self.token_scores, self.token_predictions, \
                    self.token_probabilities, self.sentence_probabilities, \
                    self.attention_weights = multi_head_attention_with_scores_from_separate_heads(
                        inputs=lstm_outputs,
                        initializer=self.initializer,
                        attention_activation=self.config["attention_activation"],
                        num_sentence_labels=len(self.label2id_sent),
                        num_heads=len(self.label2id_tok),
                        is_training=self.is_training,
                        dropout=self.config["dropout_attention"],
                        sentence_lengths=self.sentence_lengths,
                        normalize_sentence=self.config["normalize_sentence"],
                        token_scoring_method=self.config["token_scoring_method"],
                        scoring_activation=scoring_activation,
                        separate_heads=self.config["separate_heads"])
            elif self.config["model_type"] == "single_head_attention_multiple_transformations":
                self.sentence_scores, self.sentence_predictions, \
                    self.token_scores, self.token_predictions, \
                    self.token_probabilities, self.sentence_probabilities, \
                    self.attention_weights = single_head_attention_multiple_transformations(
                        inputs=lstm_outputs,
                        initializer=self.initializer,
                        attention_activation=self.config["attention_activation"],
                        num_sentence_labels=len(self.label2id_sent),
                        num_heads=len(self.label2id_tok),
                        sentence_lengths=self.sentence_lengths,
                        token_scoring_method=self.config["token_scoring_method"],
                        scoring_activation=scoring_activation,
                        how_to_compute_attention=self.config["how_to_compute_attention"],
                        separate_heads=self.config["separate_heads"])
            elif self.config["model_type"] == "variant_1":
                self.sentence_scores, self.sentence_predictions, \
                    self.token_scores, self.token_predictions, \
                    self.token_probabilities, self.sentence_probabilities, \
                    self.attention_weights = variant_1(
                        inputs=lstm_outputs,
                        initializer=self.initializer,
                        attention_activation=self.config["attention_activation"],
                        num_sentence_labels=len(self.label2id_sent),
                        num_heads=len(self.label2id_tok),
                        hidden_units=self.config["hidden_layer_size"],
                        sentence_lengths=self.sentence_lengths,
                        scoring_activation=scoring_activation,
                        token_scoring_method=self.config["token_scoring_method"],
                        use_inputs_instead_values=self.config["use_inputs_instead_values"],
                        separate_heads=self.config["separate_heads"])
            elif self.config["model_type"] == "variant_2":
                self.sentence_scores, self.sentence_predictions, \
                    self.token_scores, self.token_predictions, \
                    self.token_probabilities, self.sentence_probabilities, \
                    self.attention_weights = variant_2(
                        inputs=lstm_outputs,
                        initializer=self.initializer,
                        attention_activation=self.config["attention_activation"],
                        num_sentence_labels=len(self.label2id_sent),
                        num_heads=len(self.label2id_tok),
                        hidden_units=self.config["hidden_layer_size"],
                        sentence_lengths=self.sentence_lengths,
                        scoring_activation=scoring_activation,
                        use_inputs_instead_values=self.config["use_inputs_instead_values"],
                        separate_heads=self.config["separate_heads"])
            elif self.config["model_type"] == "variant_3":
                self.sentence_scores, self.sentence_predictions, \
                    self.token_scores, self.token_predictions, \
                    self.token_probabilities, self.sentence_probabilities, \
                    self.attention_weights = variant_3(
                        inputs=lstm_outputs,
                        initializer=self.initializer,
                        attention_activation=self.config["attention_activation"],
                        num_sentence_labels=len(self.label2id_sent),
                        num_heads=len(self.label2id_tok),
                        attention_size=self.config["attention_evidence_size"],
                        sentence_lengths=self.sentence_lengths,
                        scoring_activation=scoring_activation,
                        separate_heads=self.config["separate_heads"])
            elif self.config["model_type"] == "variant_4":
                self.sentence_scores, self.sentence_predictions, \
                    self.token_scores, self.token_predictions, \
                    self.token_probabilities, self.sentence_probabilities, \
                    self.attention_weights = variant_4(
                        inputs=lstm_outputs,
                        initializer=self.initializer,
                        attention_activation=self.config["attention_activation"],
                        num_sentence_labels=len(self.label2id_sent),
                        num_heads=len(self.label2id_tok),
                        hidden_units=self.config["hidden_layer_size"],
                        sentence_lengths=self.sentence_lengths,
                        scoring_activation=scoring_activation,
                        token_scoring_method=self.config["token_scoring_method"],
                        use_inputs_instead_values=self.config["use_inputs_instead_values"],
                        separate_heads=self.config["separate_heads"])
            elif self.config["model_type"] == "variant_5":
                self.sentence_scores, self.sentence_predictions, \
                    self.token_scores, self.token_predictions, \
                    self.token_probabilities, self.sentence_probabilities, \
                    self.attention_weights = variant_5(
                        inputs=lstm_outputs,
                        initializer=self.initializer,
                        attention_activation=self.config["attention_activation"],
                        num_sentence_labels=len(self.label2id_sent),
                        num_heads=len(self.label2id_tok),
                        hidden_units=self.config["hidden_layer_size"],
                        sentence_lengths=self.sentence_lengths,
                        scoring_activation=scoring_activation,
                        token_scoring_method=self.config["token_scoring_method"],
                        use_inputs_instead_values=self.config["use_inputs_instead_values"],
                        separate_heads=self.config["separate_heads"])
            elif self.config["model_type"] == "variant_6":
                self.sentence_scores, self.sentence_predictions, \
                    self.token_scores, self.token_predictions, \
                    self.token_probabilities, self.sentence_probabilities, \
                    self.attention_weights = variant_6(
                        inputs=lstm_outputs,
                        initializer=self.initializer,
                        attention_activation=self.config["attention_activation"],
                        num_sentence_labels=len(self.label2id_sent),
                        num_heads=len(self.label2id_tok),
                        hidden_units=self.config["hidden_layer_size"],
                        scoring_activation=scoring_activation,
                        token_scoring_method=self.config["token_scoring_method"],
                        separate_heads=self.config["separate_heads"])
            else:
                raise ValueError("Unknown/unsupported model type: %s"
                                 % self.config["model_type"])

            # Include a token-level loss (for sequence labelling).
            if self.config["word_objective_weight"] > 0:
                if self.config["token_labels_available"]:
                    word_objective_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=self.token_scores, labels=tf.cast(self.word_labels, tf.int32))
                    word_objective_loss = tf.where(
                        tf.sequence_mask(self.sentence_lengths),
                        word_objective_loss, tf.zeros_like(word_objective_loss))
                    self.loss += self.config["word_objective_weight"] * tf.reduce_sum(
                        self.word_objective_weights * word_objective_loss)
                else:
                    raise ValueError(
                        "No token labels available! You cannot supervise on the token-level"
                        " so please change \"word_objective_weight\" to 0"
                        " or provide token-annotated files.")

            # Include a sentence-level loss (for sentence classification).
            if self.config["sentence_objective_weight"] > 0:
                sentence_objective_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.sentence_scores, labels=tf.cast(self.sentence_labels, tf.int32))
                self.loss += self.config["sentence_objective_weight"] * tf.reduce_sum(sentence_objective_loss)

            # Mask the token scores that do not fall in the range of the true sentence length.
            # Do this for each head (change shape from [B, M] to [B, M, num_heads]).
            tiled_sentence_lengths = tf.tile(
                input=tf.expand_dims(
                    tf.sequence_mask(self.sentence_lengths), axis=-1),
                multiples=[1, 1, len(self.label2id_tok)])

            self.token_probabilities = tf.where(
                tiled_sentence_lengths,
                self.token_probabilities,
                tf.zeros_like(self.token_probabilities))

            if self.config["attention_objective_weight"] > 0.0:
                attention_loss = compute_attention_loss(
                    self.token_probabilities,
                    self.sentence_labels,
                    num_sent_labels=len(self.label2id_sent),
                    num_tok_labels=len(self.label2id_tok),
                    approach=self.config["aux_loss_approach"],
                    compute_pairwise=self.config["compute_pairwise_attention"])
                self.loss += self.config["attention_objective_weight"] * tf.reduce_sum(attention_loss)

            # Apply a gap-distance loss.
            if self.config["gap_objective_weight"] > 0.0:
                gap_distance_loss = compute_gap_distance_loss(
                    self.token_probabilities,
                    self.sentence_labels,
                    num_sent_labels=len(self.label2id_sent),
                    num_tok_labels=len(self.label2id_tok),
                    minimum_gap_distance=self.config["minimum_gap_distance"],
                    approach=self.config["aux_loss_approach"],
                    type_distance=self.config["type_distance"])
                self.loss += self.config["gap_objective_weight"] * tf.reduce_sum(gap_distance_loss)

        # Include a word-based language modelling loss, LMw.
        if self.config["lm_cost_lstm_gamma"] > 0.0:
            self.loss += self.config["lm_cost_lstm_gamma"] * self.construct_lm_cost(
                input_tensor_fw=lstm_outputs_fw,
                input_tensor_bw=lstm_outputs_bw,
                sentence_lengths=self.sentence_lengths,
                target_ids=self.word_ids,
                lm_cost_type="separate",
                name="lm_cost_lstm_separate")

        if self.config["lm_cost_joint_lstm_gamma"] > 0.0:
            self.loss += self.config["lm_cost_joint_lstm_gamma"] * self.construct_lm_cost(
                input_tensor_fw=lstm_outputs_fw,
                input_tensor_bw=lstm_outputs_bw,
                sentence_lengths=self.sentence_lengths,
                target_ids=self.word_ids,
                lm_cost_type="joint",
                name="lm_cost_lstm_joint")

        self.train_op = self.construct_optimizer(
            opt_strategy=self.config["opt_strategy"],
            loss=self.loss,
            learning_rate=self.learning_rate,
            clip=self.config["clip"])
        print("Notwork built.")

    def construct_lm_cost(
            self, input_tensor_fw, input_tensor_bw,
            sentence_lengths, target_ids, lm_cost_type, name):
        """
        Constructs the char/word-based language modelling objective.
        """
        with tf.variable_scope(name):
            lm_cost_max_vocab_size = min(
                len(self.word2id), self.config["lm_cost_max_vocab_size"])
            target_ids = tf.where(
                tf.greater_equal(target_ids, lm_cost_max_vocab_size - 1),
                x=(lm_cost_max_vocab_size - 1) + tf.zeros_like(target_ids),
                y=target_ids)
            cost = 0.0
            if lm_cost_type == "separate":
                lm_cost_fw_mask = tf.sequence_mask(
                    sentence_lengths, maxlen=tf.shape(target_ids)[1])[:, 1:]
                lm_cost_bw_mask = tf.sequence_mask(
                    sentence_lengths, maxlen=tf.shape(target_ids)[1])[:, :-1]
                lm_cost_fw = self._construct_lm_cost(
                    input_tensor_fw[:, :-1, :],
                    lm_cost_max_vocab_size,
                    lm_cost_fw_mask,
                    target_ids[:, 1:],
                    name=name + "_fw")
                lm_cost_bw = self._construct_lm_cost(
                    input_tensor_bw[:, 1:, :],
                    lm_cost_max_vocab_size,
                    lm_cost_bw_mask,
                    target_ids[:, :-1],
                    name=name + "_bw")
                cost += lm_cost_fw + lm_cost_bw
            elif lm_cost_type == "joint":
                joint_input_tensor = tf.concat(
                    [input_tensor_fw[:, :-2, :], input_tensor_bw[:, 2:, :]], axis=-1)
                lm_cost_mask = tf.sequence_mask(
                    sentence_lengths, maxlen=tf.shape(target_ids)[1])[:, 1:-1]
                cost += self._construct_lm_cost(
                    joint_input_tensor,
                    lm_cost_max_vocab_size,
                    lm_cost_mask,
                    target_ids[:, 1:-1],
                    name=name + "_joint")
            else:
                raise ValueError("Unknown lm_cost_type: %s." % lm_cost_type)
            return cost

    def _construct_lm_cost(
            self, input_tensor, lm_cost_max_vocab_size,
            lm_cost_mask, target_ids, name):
        with tf.variable_scope(name):
            lm_cost_hidden_layer = tf.layers.dense(
                inputs=input_tensor, units=self.config["lm_cost_hidden_layer_size"],
                activation=tf.tanh, kernel_initializer=self.initializer)
            lm_cost_output = tf.layers.dense(
                inputs=lm_cost_hidden_layer, units=lm_cost_max_vocab_size,
                kernel_initializer=self.initializer)
            lm_cost_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=lm_cost_output, labels=target_ids)
            lm_cost_loss = tf.where(lm_cost_mask, lm_cost_loss, tf.zeros_like(lm_cost_loss))
            return tf.reduce_sum(lm_cost_loss)

    @staticmethod
    def construct_optimizer(opt_strategy, loss, learning_rate, clip):
        """
        Applies an optimization strategy to minimize the loss.
        """
        if opt_strategy == "adadelta":
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        elif opt_strategy == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif opt_strategy == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        else:
            raise ValueError("Unknown optimisation strategy: %s." % opt_strategy)

        if clip > 0.0:
            grads, vs = zip(*optimizer.compute_gradients(loss))
            grads, gnorm = tf.clip_by_global_norm(grads, clip)
            train_op = optimizer.apply_gradients(zip(grads, vs))
        else:
            train_op = optimizer.minimize(loss)
        return train_op

    def preload_word_embeddings(self, embedding_path):
        """
        Load the word embeddings in advance to get a feel
        of the proportion of singletons in the dataset.
        """
        loaded_embeddings = set()
        embedding_matrix = self.session.run(self.word_embeddings)
        with open(embedding_path) as f:
            for line in f:
                line_parts = line.strip().split()
                if len(line_parts) <= 2:
                    continue
                w = line_parts[0]
                if self.config["lowercase"]:
                    w = w.lower()
                if self.config["replace_digits"]:
                    w = re.sub(r'\d', '0', w)
                if w in self.word2id and w not in loaded_embeddings:
                    word_id = self.word2id[w]
                    embedding = numpy.array(line_parts[1:])
                    embedding_matrix[word_id] = embedding
                    loaded_embeddings.add(w)
        self.session.run(self.word_embeddings.assign(embedding_matrix))
        print("No. of pre-loaded embeddings: %d." % len(loaded_embeddings))

    @staticmethod
    def translate2id(
            token, token2id, unk_token=None, lowercase=False,
            replace_digits=False, singletons=None, singletons_prob=0.0):
        """
        Maps each token/character to its index.
        """
        if lowercase:
            token = token.lower()
        if replace_digits:
            token = re.sub(r'\d', '0', token)

        if singletons and token in singletons \
                and token in token2id and unk_token \
                and numpy.random.uniform() < singletons_prob:
            token_id = token2id[unk_token]
        elif token in token2id:
            token_id = token2id[token]
        elif unk_token:
            token_id = token2id[unk_token]
        else:
            raise ValueError("Unable to handle value, no UNK token: %s." % token)
        return token_id

    def create_input_dictionary_for_batch(self, batch, is_training, learning_rate):
        """
        Creates the dictionary fed to the the TF model.
        """
        sentence_lengths = numpy.array([len(sentence.tokens) for sentence in batch])
        max_sentence_length = sentence_lengths.max()
        max_word_length = numpy.array(
            [numpy.array([len(token.value) for token in sentence.tokens]).max()
             for sentence in batch]).max()

        if 0 < self.config["allowed_word_length"] < max_word_length:
            max_word_length = min(max_word_length, self.config["allowed_word_length"])

        word_ids = numpy.zeros(
            (len(batch), max_sentence_length), dtype=numpy.int32)
        char_ids = numpy.zeros(
            (len(batch), max_sentence_length, max_word_length), dtype=numpy.int32)
        word_lengths = numpy.zeros(
            (len(batch), max_sentence_length), dtype=numpy.int32)
        word_labels = numpy.zeros(
            (len(batch), max_sentence_length), dtype=numpy.float32)
        sentence_labels = numpy.zeros(
            (len(batch)), dtype=numpy.float32)
        word_objective_weights = numpy.zeros(
            (len(batch), max_sentence_length), dtype=numpy.float32)
        sentence_objective_weights = numpy.zeros((len(batch)), dtype=numpy.float32)

        # A proportion of the singletons are assigned to UNK (do this just for training).
        singletons = self.singletons if is_training else None
        singletons_prob = self.config["singletons_prob"] if is_training else 0.0

        for i, sentence in enumerate(batch):
            sentence_labels[i] = sentence.label_sent

            if sentence_labels[i] != 0:
                if self.config["sentence_objective_weights_non_default"] > 0.0:
                    sentence_objective_weights[i] = self.config[
                        "sentence_objective_weights_non_default"]
                else:
                    sentence_objective_weights[i] = 1.0
            else:
                sentence_objective_weights[i] = 1.0

            for j, token in enumerate(sentence.tokens):
                word_ids[i][j] = self.translate2id(
                    token=token.value,
                    token2id=self.word2id,
                    unk_token=self.UNK,
                    lowercase=self.config["lowercase"],
                    replace_digits=self.config["replace_digits"],
                    singletons=singletons,
                    singletons_prob=singletons_prob)
                word_labels[i][j] = token.label_tok
                word_lengths[i][j] = len(token.value)
                for k in range(min(len(token.value), max_word_length)):
                    char_ids[i][j][k] = self.translate2id(
                        token=token.value[k],
                        token2id=self.char2id,
                        unk_token=self.CUNK)
                if token.enable_supervision is True:
                    word_objective_weights[i][j] = 1.0

        input_dictionary = {
            self.word_ids: word_ids,
            self.char_ids: char_ids,
            self.sentence_lengths: sentence_lengths,
            self.word_lengths: word_lengths,
            self.sentence_labels: sentence_labels,
            self.word_labels: word_labels,
            self.word_objective_weights: word_objective_weights,
            self.learning_rate: learning_rate,
            self.is_training: is_training}
        return input_dictionary

    def process_batch(self, batch, is_training, learning_rate):
        """
        Processes a batch of sentences.
        :param batch: a set of sentences of size "max_batch_size".
        :param is_training: whether the current batch is a training instance or not.
        :param learning_rate: the pace at which learning should be performed.
        :return: the cost, the sentence predictions, the sentence label distribution,
        the token predictions and the token label distribution.
        """
        feed_dict = self.create_input_dictionary_for_batch(batch, is_training, learning_rate)
        cost, sentence_pred, sentence_prob, token_pred, token_prob = self.session.run(
            [self.loss, self.sentence_predictions, self.sentence_probabilities,
             self.token_predictions, self.token_probabilities] +
            ([self.train_op] if is_training else []), feed_dict=feed_dict)[:5]
        return cost, sentence_pred, sentence_prob, token_pred, token_prob

    def initialize_session(self):
        """
        Initializes a tensorflow session and sets the random seed.
        """
        tf.set_random_seed(self.config["random_seed"])
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = self.config["tf_allow_growth"]
        session_config.gpu_options.per_process_gpu_memory_fraction = self.config[
            "tf_per_process_gpu_memory_fraction"]
        self.session = tf.Session(config=session_config)
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)

    @staticmethod
    def get_parameter_count():
        """
        Counts the total number of parameters.
        """
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters

    def get_parameter_count_without_word_embeddings(self):
        """
        Counts the number of parameters without those introduced by word embeddings.
        """
        shape = self.word_embeddings.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        return self.get_parameter_count() - variable_parameters

    def save(self, filename):
        """
        Saves a trained model to the path in filename.
        """
        dump = dict()
        dump["config"] = self.config
        dump["label2id_sent"] = self.label2id_sent
        dump["label2id_tok"] = self.label2id_tok
        dump["UNK"] = self.UNK
        dump["CUNK"] = self.CUNK
        dump["word2id"] = self.word2id
        dump["char2id"] = self.char2id
        dump["singletons"] = self.singletons

        dump["params"] = {}
        for variable in tf.global_variables():
            assert (
                variable.name not in dump["params"]), \
                "Error: variable with this name already exists: %s." % variable.name
            dump["params"][variable.name] = self.session.run(variable)
        with open(filename, 'wb') as f:
            pickle.dump(dump, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename, new_config=None):
        """
        Loads a pre-trained MHAL model.
        """
        with open(filename, 'rb') as f:
            dump = pickle.load(f)
            dump["config"]["save"] = None

            # Use the saved config, except for values that are present in the new config.
            if new_config:
                for key in new_config:
                    dump["config"][key] = new_config[key]

            labeler = Model(dump["config"], dump["label2id_sent"], dump["label2id_tok"])
            labeler.UNK = dump["UNK"]
            labeler.CUNK = dump["CUNK"]
            labeler.word2id = dump["word2id"]
            labeler.char2id = dump["char2id"]
            labeler.singletons = dump["singletons"]

            labeler.construct_network()
            labeler.initialize_session()

            labeler.load_params(filename)

            return labeler

    def load_params(self, filename):
        """
        Loads the parameters of a trained model.
        """
        with open(filename, 'rb') as f:
            dump = pickle.load(f)

            for variable in tf.global_variables():
                assert (variable.name in dump["params"]), \
                    "Variable not in dump: %s." % variable.name
                assert (variable.shape == dump["params"][variable.name].shape), \
                    "Variable shape not as expected: %s, of shape %s. %s" % (
                        variable.name, str(variable.shape),
                        str(dump["params"][variable.name].shape))
                value = numpy.asarray(dump["params"][variable.name])
                self.session.run(variable.assign(value))

