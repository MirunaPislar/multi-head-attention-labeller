from math import ceil
from modules import single_head_attention_binary_labels
from modules import multi_head_attention_with_scores_from_shared_heads
from modules import multi_head_attention_with_scores_from_separate_heads
from modules import single_head_attention_multiple_labels
from modules import single_head_attention_multiple_transformations
from modules import baseline_lstm_last_contexts
from modules import compute_attention_loss
from modules import compute_gap_distance_loss
from variants import variant_1, variant_2, variant_3, variant_4, variant_5, variant_6
import collections
import numpy
import pickle
import re
import tensorflow as tf


class Model(object):

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
        self.sentence_objective_weights = None
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
            with open(embedding_path, 'r') as f:
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
        self.word_ids = tf.placeholder(tf.int32, [None, None], name="word_ids")
        self.char_ids = tf.placeholder(tf.int32, [None, None, None], name="char_ids")
        self.sentence_lengths = tf.placeholder(tf.int32, [None], name="sentence_lengths")
        self.word_lengths = tf.placeholder(tf.int32, [None, None], name="word_lengths")
        self.sentence_labels = tf.placeholder(tf.float32, [None], name="sentence_labels")
        self.word_labels = tf.placeholder(tf.float32, [None, None], name="word_labels")

        self.word_objective_weights = tf.placeholder(
            tf.float32, [None, None], name="word_objective_weights")
        self.sentence_objective_weights = tf.placeholder(
            tf.float32, [None], name="sentence_objective_weights")

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
                    shape=[char_input_tensor_shape[0] * char_input_tensor_shape[1],
                           char_input_tensor_shape[2], self.config["char_embedding_size"]])
                _word_lengths = tf.reshape(
                    self.word_lengths, shape=[char_input_tensor_shape[0] * char_input_tensor_shape[1]])

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

                # Use just the forward and the backward final character states
                _, ((_, char_output_fw), (_, char_output_bw)) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=char_lstm_cell_fw, cell_bw=char_lstm_cell_bw, inputs=char_input_tensor,
                    sequence_length=_word_lengths, dtype=tf.float32, time_major=False)

                char_output_tensor = tf.concat([char_output_fw, char_output_bw], axis=-1)
                char_output_tensor = tf.reshape(
                    char_output_tensor,
                    shape=[char_input_tensor_shape[0], char_input_tensor_shape[1],
                           2 * self.config["char_recurrent_size"]])

                # This is the character language model loss, L_char
                if self.config["lmcost_char_gamma"] > 0.0:
                    self.loss += self.config["lmcost_char_gamma"] * \
                                 self.construct_lmcost(
                                     input_tensor_fw=char_output_tensor,
                                     input_tensor_bw=char_output_tensor,
                                     sentence_lengths=self.sentence_lengths,
                                     target_ids=self.word_ids,
                                     lmcost_type="separate",
                                     name="lmcost_char_separate")

                if self.config["lmcost_joint_char_gamma"] > 0.0:
                    self.loss += self.config["lmcost_joint_char_gamma"] * \
                                 self.construct_lmcost(
                                     input_tensor_fw=char_output_tensor,
                                     input_tensor_bw=char_output_tensor,
                                     sentence_lengths=self.sentence_lengths,
                                     target_ids=self.word_ids,
                                     lmcost_type="joint",
                                     name="lmcost_char_joint")

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

        # The states are concatenated at every token position.
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

            # Token-level loss
            word_objective_loss = tf.square(self.token_scores - self.word_labels)
            word_objective_loss = tf.where(
                tf.sequence_mask(self.sentence_lengths),
                word_objective_loss, tf.zeros_like(word_objective_loss))
            self.loss += self.config["word_objective_weight"] * tf.reduce_sum(
                self.word_objective_weights * word_objective_loss)

            # Sentence-level loss
            sentence_objective_loss = tf.square(self.sentence_scores - self.sentence_labels)
            self.loss += self.config["sentence_objective_weight"] * tf.reduce_sum(
                self.sentence_objective_weights * sentence_objective_loss)

            # Attention-level loss
            if self.config["attention_objective_weight"] > 0.0:
                self.loss += self.config["attention_objective_weight"] * (
                    tf.reduce_sum(
                        self.sentence_objective_weights * tf.square(
                            tf.reduce_max(
                                tf.where(
                                    tf.sequence_mask(self.sentence_lengths),
                                    self.token_scores,
                                    tf.zeros_like(self.token_scores) - 1e6),
                                axis=-1) - self.sentence_labels))
                    +
                    tf.reduce_sum(
                        self.sentence_objective_weights * tf.square(
                            tf.reduce_min(
                                tf.where(
                                    tf.sequence_mask(self.sentence_lengths),
                                    self.token_scores,
                                    tf.zeros_like(self.token_scores) + 1e6),
                                axis=-1) - 0.0)))
        else:
            scoring_activation = None
            if self.config["scoring_activation"]:
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
                raise ValueError("Unknown/unsupported model type: %s" % self.config["model_type"])

            # Token-level loss
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

            # Sentence-level loss
            if self.config["sentence_objective_weight"] > 0:
                sentence_objective_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.sentence_scores, labels=tf.cast(self.sentence_labels, tf.int32))
                self.loss += self.config["sentence_objective_weight"] * tf.reduce_sum(
                    self.sentence_objective_weights * sentence_objective_loss)

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

        # Token-level language modelling objective
        if self.config["lmcost_lstm_gamma"] > 0.0:
            self.loss += self.config["lmcost_lstm_gamma"] * self.construct_lmcost(
                input_tensor_fw=lstm_outputs_fw,
                input_tensor_bw=lstm_outputs_bw,
                sentence_lengths=self.sentence_lengths,
                target_ids=self.word_ids,
                lmcost_type="separate",
                name="lmcost_lstm_separate")

        if self.config["lmcost_joint_lstm_gamma"] > 0.0:
            self.loss += self.config["lmcost_joint_lstm_gamma"] * self.construct_lmcost(
                input_tensor_fw=lstm_outputs_fw,
                input_tensor_bw=lstm_outputs_bw,
                sentence_lengths=self.sentence_lengths,
                target_ids=self.word_ids,
                lmcost_type="joint",
                name="lmcost_lstm_joint")

        self.train_op = self.construct_optimizer(
            opt_strategy=self.config["opt_strategy"],
            loss=self.loss,
            learning_rate=self.learning_rate,
            clip=self.config["clip"])
        print("Notwork built.")

    def construct_lmcost(self, input_tensor_fw, input_tensor_bw,
                         sentence_lengths, target_ids, lmcost_type, name):
        with tf.variable_scope(name):
            lmcost_max_vocab_size = min(
                len(self.word2id), self.config["lmcost_max_vocab_size"])
            target_ids = tf.where(
                tf.greater_equal(target_ids, lmcost_max_vocab_size - 1),
                x=(lmcost_max_vocab_size - 1) + tf.zeros_like(target_ids),
                y=target_ids)
            cost = 0.0
            if lmcost_type == "separate":
                lmcost_fw_mask = tf.sequence_mask(
                    sentence_lengths, maxlen=tf.shape(target_ids)[1])[:, 1:]
                lmcost_bw_mask = tf.sequence_mask(
                    sentence_lengths, maxlen=tf.shape(target_ids)[1])[:, :-1]
                lmcost_fw = self._construct_lmcost(
                    input_tensor_fw[:, :-1, :],
                    lmcost_max_vocab_size,
                    lmcost_fw_mask,
                    target_ids[:, 1:],
                    name=name + "_fw")
                lmcost_bw = self._construct_lmcost(
                    input_tensor_bw[:, 1:, :],
                    lmcost_max_vocab_size,
                    lmcost_bw_mask,
                    target_ids[:, :-1],
                    name=name + "_bw")
                cost += lmcost_fw + lmcost_bw
            elif lmcost_type == "joint":
                joint_input_tensor = tf.concat(
                    [input_tensor_fw[:, :-2, :], input_tensor_bw[:, 2:, :]], axis=-1)
                lmcost_mask = tf.sequence_mask(
                    sentence_lengths, maxlen=tf.shape(target_ids)[1])[:, 1:-1]
                cost += self._construct_lmcost(
                    joint_input_tensor,
                    lmcost_max_vocab_size,
                    lmcost_mask,
                    target_ids[:, 1:-1],
                    name=name + "_joint")
            else:
                raise ValueError("Unknown lmcost_type: " + str(lmcost_type))
            return cost

    def _construct_lmcost(self, input_tensor, lmcost_max_vocab_size, lmcost_mask, target_ids, name):
        with tf.variable_scope(name):
            lmcost_hidden_layer = tf.layers.dense(
                inputs=input_tensor, units=self.config["lmcost_hidden_layer_size"],
                activation=tf.tanh, kernel_initializer=self.initializer)
            lmcost_output = tf.layers.dense(
                inputs=lmcost_hidden_layer, units=lmcost_max_vocab_size,
                activation=None, kernel_initializer=self.initializer)
            lmcost_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=lmcost_output, labels=target_ids)
            lmcost_loss = tf.where(lmcost_mask, lmcost_loss, tf.zeros_like(lmcost_loss))
            return tf.reduce_sum(lmcost_loss)

    @staticmethod
    def construct_optimizer(opt_strategy, loss, learning_rate, clip):
        if opt_strategy == "adadelta":
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        elif opt_strategy == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif opt_strategy == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        else:
            raise ValueError("Unknown optimisation strategy: " + str(opt_strategy))

        if clip > 0.0:
            grads, vs = zip(*optimizer.compute_gradients(loss))
            grads, gnorm = tf.clip_by_global_norm(grads, clip)
            train_op = optimizer.apply_gradients(zip(grads, vs))
        else:
            train_op = optimizer.minimize(loss)
        return train_op

    def preload_word_embeddings(self, embedding_path):
        loaded_embeddings = set()
        embedding_matrix = self.session.run(self.word_embeddings)
        with open(embedding_path, "r") as f:
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
        print("n_preloaded_embeddings: " + str(len(loaded_embeddings)))

    @staticmethod
    def translate2id(token, token2id, unk_token=None,
                     lowercase=False, replace_digits=False,
                     singletons=None, singletons_prob=0.0):
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
            raise ValueError("Unable to handle value, no UNK token: " + str(token))
        return token_id

    def create_input_dictionary_for_batch(self, batch, is_training, learning_rate):
        sentence_lengths = numpy.array([len(sentence.tokens) for sentence in batch])
        max_sentence_length = sentence_lengths.max()
        max_word_length = numpy.array(
            [numpy.array([len(token.value) for token in sentence.tokens]).max() for sentence in batch]).max()

        if 0 < self.config["allowed_word_length"] < max_word_length:
            max_word_length = min(max_word_length, self.config["allowed_word_length"])

        word_ids = numpy.zeros((len(batch), max_sentence_length), dtype=numpy.int32)
        char_ids = numpy.zeros((len(batch), max_sentence_length, max_word_length), dtype=numpy.int32)
        word_lengths = numpy.zeros((len(batch), max_sentence_length), dtype=numpy.int32)
        word_labels = numpy.zeros((len(batch), max_sentence_length), dtype=numpy.float32)
        sentence_labels = numpy.zeros((len(batch)), dtype=numpy.float32)
        word_objective_weights = numpy.zeros((len(batch), max_sentence_length), dtype=numpy.float32)
        sentence_objective_weights = numpy.zeros((len(batch)), dtype=numpy.float32)

        # A proportion of the sentences should be assigned to UNK. We do this just for training.
        singletons = self.singletons if is_training else None
        singletons_prob = self.config["singletons_prob"] if is_training else 0.0

        for i, sentence in enumerate(batch):
            sentence_labels[i] = sentence.label_sent
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
                # if len(batch[i][j]) == 2 or (len(batch[i][j]) >= 3 and batch[i][j][1] == "T"):
                word_objective_weights[i][j] = 1.0
            # if len(batch[i][j]) == 2 or (len(batch[i][j]) >= 3 and batch[i][0][1] == "S") \
            #       or self.config["sentence_objective_persistent"]:
            sentence_objective_weights[i] = 1.0

        input_dictionary = {
            self.word_ids: word_ids,
            self.char_ids: char_ids,
            self.sentence_lengths: sentence_lengths,
            self.word_lengths: word_lengths,
            self.sentence_labels: sentence_labels,
            self.word_labels: word_labels,
            self.word_objective_weights: word_objective_weights,
            self.sentence_objective_weights: sentence_objective_weights,
            self.learning_rate: learning_rate,
            self.is_training: is_training}
        return input_dictionary

    def process_batch(self, batch, is_training, learning_rate):
        feed_dict = self.create_input_dictionary_for_batch(batch, is_training, learning_rate)
        cost, sentence_scores, token_scores, sentence_pred, token_pred, \
        token_proba, sent_proba, attention_weights = self.session.run(
            [self.loss, self.sentence_scores, self.token_scores,
             self.sentence_predictions, self.token_predictions,
             self.token_probabilities, self.sentence_probabilities, self.attention_weights] +
            ([self.train_op] if is_training else []), feed_dict=feed_dict)[:8]
        # print("Sent lengths: ", sent_lengths.shape)
        # print("M = self.token_probabilities, of shape ", m.shape)
        # print("N = weighted_sum_representation, of shape ", n.shape)
        # print("P = processed_tensor, of shape ", p.shape)
        # print("Sent lengths = ", sent_lengths)
        # print("*" * 50, "\nM =\n", "\n\n".join(["\n".join([str(elem) for elem in e]) for e in m]))
        # print("*" * 50, "\nN =\n", "\n\n".join(["\n".join([str(elem) for elem in e]) for e in
        #                                         [n[0], n[1], n[-2], n[-1]]]))
        # print("*" * 50, "\nP =\n", "\n\n".join(["\n".join([str(elem) for elem in e]) for e in p]))

        return cost, sentence_scores, token_scores, sentence_pred, token_pred, \
            token_proba, sent_proba, attention_weights

    def initialize_session(self):
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
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters

    def get_parameter_count_without_word_embeddings(self):
        shape = self.word_embeddings.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        return self.get_parameter_count() - variable_parameters

    def save(self, filename):
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
                "Error: variable with this name already exists" + str(variable.name)
            dump["params"][variable.name] = self.session.run(variable)
        with open(filename, 'wb') as f:
            pickle.dump(dump, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename, new_config=None):
        with open(filename, 'rb') as f:
            dump = pickle.load(f)

            # for safety, so we don't overwrite old models
            dump["config"]["save"] = None

            # we use the saved config, except for values that are present in the new config
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
        with open(filename, 'rb') as f:
            dump = pickle.load(f)

            for variable in tf.global_variables():
                assert (variable.name in dump["params"]), "Variable not in dump: " + str(variable.name)
                assert (variable.shape == dump["params"][variable.name].shape), \
                    "Variable shape not as expected: " + str(variable.name) \
                    + " " + str(variable.shape) + " " \
                    + str(dump["params"][variable.name].shape)
                value = numpy.asarray(dump["params"][variable.name])
                self.session.run(variable.assign(value))

