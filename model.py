import collections
import math
import numpy
import pickle
import re
import tensorflow as tf


class MLTModel:
    def __init__(self, config):
        self.config = config
        self.UNK = "<unk>"
        self.CUNK = "<cunk>"
        self.word2id = None
        self.char2id = None
        self.singletons = None
        self.sent_label2id = None
        self.token_label2id = None
        self.num_heads = None

    def build_vocabs(self, data_train, data_dev, data_test, embedding_path=None):
        data_source = list(data_train)
        if self.config["vocab_include_devtest"]:
            if data_dev is not None:
                data_source += data_dev
            if data_test is not None:
                data_source += data_test

        token_labels = set()
        sent_labels = set()
        char_counter = collections.Counter()
        word_counter = collections.Counter()
        for sentence in data_source:
            if self.config["binary_labels"]:
                the_sentence = sentence
            else:
                the_sentence = sentence[:-1]
            for word in the_sentence:
                w = word[0]
                char_counter.update(w)
                if self.config["lowercase"]:
                    w = w.lower()
                if self.config["replace_digits"]:
                    w = re.sub(r'\d', '0', w)
                word_counter[w] += 1
                token_labels.add(word[1])
            if not self.config["binary_labels"]:
                sent_labels.add(sentence[-1][1])

        self.char2id = collections.OrderedDict([(self.CUNK, 0)])
        for char, count in char_counter.most_common():
            if char not in self.char2id:
                self.char2id[char] = len(self.char2id)

        self.word2id = collections.OrderedDict([(self.UNK, 0)])
        for word, count in word_counter.most_common():
            if self.config["min_word_freq"] <= 0 or count >= self.config["min_word_freq"]:
                if word not in self.word2id:
                    self.word2id[word] = len(self.word2id)

        # Get the L unique sentiment labels and map them to an index in [0, L).
        # If the default label is specified and among the discovered sentence labels,
        # it will receive the index 0. All other labels get the index defined by their (string) order.
        if not self.config["binary_labels"]:
            if self.config["default_label"] and self.config["default_label"] in sent_labels:
                sorted_labels = sorted(list(sent_labels.difference(self.config["default_label"])))
                self.sent_label2id = {label: index + 1 for index, label in enumerate(sorted_labels)}
                self.sent_label2id[self.config["default_label"]] = 0
            else:
                sorted_labels = sorted(list(sent_labels))
                self.sent_label2id = {label: index for index, label in enumerate(sorted_labels)}
        print("Sent label2id: ", self.sent_label2id)

        # Define the number of heads = number of unique token labels
        self.num_heads = len(token_labels)
        print("Number of heads = ", self.num_heads)

        # Get the T unique token labels and map them to an index in [0, T).
        # If the default label is specified and among the discovered token labels,
        # it will receive the index 0. All other labels get the index defined by their (string) order.
        if not self.config["binary_labels"]:
            if self.config["default_label"] and self.config["default_label"] in token_labels:
                sorted_labels = sorted(list(token_labels.difference(self.config["default_label"])))
                self.token_label2id = {label: index + 1 for index, label in enumerate(sorted_labels)}
                self.token_label2id[self.config["default_label"]] = 0
            else:
                sorted_labels = sorted(list(token_labels))
                self.token_label2id = {label: index for index, label in enumerate(sorted_labels)}
        print("Token_label2id: ", self.token_label2id)

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

        print("n_words: " + str(len(self.word2id)))
        print("n_chars: " + str(len(self.char2id)))
        print("n_singletons: " + str(len(self.singletons)))

    def construct_network(self):
        self.word_ids = tf.placeholder(tf.int32, [None, None], name="word_ids")
        self.char_ids = tf.placeholder(tf.int32, [None, None, None], name="char_ids")
        self.sentence_lengths = tf.placeholder(tf.int32, [None], name="sentence_lengths")
        self.word_lengths = tf.placeholder(tf.int32, [None, None], name="word_lengths")
        self.sentence_labels = tf.placeholder(tf.float32, [None, ], name="sentence_labels")
        self.word_labels = tf.placeholder(tf.float32, [None, None], name="word_labels")
        self.word_objective_weights = tf.placeholder(tf.float32, [None, None], name="word_objective_weights")
        self.sentence_objective_weights = tf.placeholder(tf.float32, [None], name="sentence_objective_weights")
        self.learningrate = tf.placeholder(tf.float32, name="learningrate")
        self.is_training = tf.placeholder(tf.int32, name="is_training")
        self.loss = 0.0

        self.initializer = None
        if self.config["initializer"] == "normal":
            self.initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
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

                s = tf.shape(char_input_tensor)
                char_input_tensor = tf.reshape(
                    char_input_tensor,
                    shape=[s[0] * s[1], s[2], self.config["char_embedding_size"]])
                _word_lengths = tf.reshape(self.word_lengths, shape=[s[0] * s[1]])

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
                    char_output_tensor, shape=[s[0], s[1], 2 * self.config["char_recurrent_size"]])

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

        self.word_representations = word_input_tensor

        dropout_input = self.config["dropout_input"] * tf.cast(self.is_training, tf.float32) \
                        + (1.0 - tf.cast(self.is_training, tf.float32))
        word_input_tensor = tf.nn.dropout(word_input_tensor, dropout_input, name="dropout_word")

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
            # Outputs containing the forward and the backward rnn output, (of shape batch_size x max_len x emb_size)
            # followed by the forward and the backward final states of bidir-rnn (of shape batch_size x emb_size).
            (lstm_outputs_fw, lstm_outputs_bw), ((_, lstm_output_fw), (_, lstm_output_bw)) = \
                tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=word_lstm_cell_fw, cell_bw=word_lstm_cell_bw, inputs=word_input_tensor,
                    sequence_length=self.sentence_lengths, dtype=tf.float32, time_major=False)

        dropout_word_lstm = self.config["dropout_word_lstm"] * tf.cast(
            self.is_training, tf.float32) + (1.0 - tf.cast(self.is_training, tf.float32))

        # HERE: what is noise_shape? What does it control?
        lstm_outputs_fw = tf.nn.dropout(    # B x Max_len x emb_size
            lstm_outputs_fw, dropout_word_lstm,
            noise_shape=tf.convert_to_tensor(
                [tf.shape(self.word_ids)[0], 1, self.config["word_recurrent_size"]], dtype=tf.int32))
        lstm_outputs_bw = tf.nn.dropout(    # B x Max_len x emb_size
            lstm_outputs_bw, dropout_word_lstm,
            noise_shape=tf.convert_to_tensor(
                [tf.shape(self.word_ids)[0], 1, self.config["word_recurrent_size"]], dtype=tf.int32))

        # Note the -1; in this way, the states are concatenated at every token position
        lstm_outputs = tf.concat([lstm_outputs_fw, lstm_outputs_bw], axis=-1)    # batch_size x max_len x 2 * emb_size

        if self.config["whidden_layer_size"] > 0:
            lstm_outputs = tf.layers.dense(
                inputs=lstm_outputs, units=self.config["whidden_layer_size"],
                activation=tf.tanh, kernel_initializer=self.initializer)    # B x M x hidden_size

        lstm_output = tf.concat([lstm_output_fw, lstm_output_bw], -1)   # B x 2 * emb_size
        lstm_output = tf.nn.dropout(lstm_output, dropout_word_lstm)     # same as above

        # =================================================================================

        # Token_scores of shape [B x M x num_heads]. Sentence_scores of shape [B x num_heads]
        multihead_token_scores, multihead_sentence_scores = self.my_multihead_attention(
            inputs=lstm_outputs, num_heads=self.num_heads, reuse=False)

        self.token_scores = multihead_token_scores
        self.sentence_scores = multihead_sentence_scores

        # Argmax over the tokens scores. Shape is [B x M].
        argmax_multihead_token_scores = tf.cast(tf.argmax(multihead_token_scores, axis=-1), tf.float32)

        # Argmax over the sentence scores. Shape is [B].
        argmax_multihead_sentence_scores = tf.cast(tf.argmax(multihead_sentence_scores, axis=-1), tf.float32)

        self.multihead_sentence_scores = argmax_multihead_sentence_scores
        self.multihead_token_scores = [tf.where(tf.sequence_mask(self.sentence_lengths),
                                                argmax_multihead_token_scores,
                                                tf.zeros_like(argmax_multihead_token_scores) - 1e6)]

        # This is the loss for the word predictions, L_tok
        self.one_hot_word_labels = tf.one_hot(tf.cast(self.word_labels, tf.int64), depth=self.num_heads)
        word_objective_loss = tf.reduce_sum(
            tf.square(multihead_token_scores - self.one_hot_word_labels), axis=-1)
        word_objective_loss = tf.where(
            tf.sequence_mask(self.sentence_lengths),
            word_objective_loss, tf.zeros_like(word_objective_loss))
        self.loss += self.config["word_objective_weight"] * tf.reduce_sum(
            self.word_objective_weights * word_objective_loss)

        if not self.config["use_one_hot"]:
            # This is the loss for the sentence predictions, L_sent
            self.loss += self.config["sentence_objective_weight"] * tf.reduce_sum(
                self.sentence_objective_weights * tf.square(
                    argmax_multihead_sentence_scores - self.sentence_labels))
        else:
            # ONE-HOT alternative for the loss for the sentence predictions, L_sent
            self.one_hot_sent_labels = tf.one_hot(tf.cast(self.sentence_labels, tf.int64), depth=self.num_heads)
            self.loss += self.config["sentence_objective_weight"] * tf.reduce_sum(
                self.sentence_objective_weights * tf.reduce_sum(tf.square(
                    multihead_sentence_scores - self.one_hot_sent_labels), axis=-1))

        max_multihead_token_scores = tf.reduce_max(
            tf.transpose(multihead_token_scores, perm=[0, 2, 1]), axis=-1)
        max_multihead_token_scores = tf.cast(
            tf.argmax(max_multihead_token_scores, axis=-1), tf.float32)
        self.mm = max_multihead_token_scores

        # Loss for attention, L_attn
        if self.config["attention_objective_weight"] > 0.0:
            self.loss += self.config["attention_objective_weight"] * (
                tf.reduce_sum(
                    self.sentence_objective_weights * tf.square(
                        max_multihead_token_scores - self.sentence_labels)))

        # =================================================================================
        """
        processed_tensor = lstm_output

        if self.config["sentence_composition"] == "attention":
            with tf.variable_scope("attention"):
                attention_evidence = tf.layers.dense(
                    inputs=lstm_outputs, units=self.config["attention_evidence_size"],
                    activation=tf.tanh, kernel_initializer=self.initializer)    # B x M x Attention_size

                attention_weights = tf.layers.dense(
                    inputs=attention_evidence, units=1,
                    activation=None, kernel_initializer=self.initializer)   # B x M x 1

                attention_weights = tf.reshape(attention_weights, shape=tf.shape(self.word_ids))    # B x M

                if self.config["attention_activation"] == "sharp":
                    attention_weights = tf.exp(attention_weights)
                elif self.config["attention_activation"] == "soft":
                    attention_weights = tf.sigmoid(attention_weights)
                elif self.config["attention_activation"] == "linear":
                    pass
                else:
                    raise ValueError("Unknown activation for attention: " + str(self.config["attention_activation"]))

                # This is the loss for the word predictions, L_tok (in the paper)
                word_objective_loss = tf.square(attention_weights - self.word_labels)
                word_objective_loss = tf.where(
                    tf.sequence_mask(self.sentence_lengths),
                    word_objective_loss, tf.zeros_like(word_objective_loss))
                self.loss += self.config["word_objective_weight"] * tf.reduce_sum(
                    self.word_objective_weights * word_objective_loss)

                self.attention_weights_unnormalised = attention_weights     # B x M
                attention_weights = tf.where(
                    tf.sequence_mask(self.sentence_lengths),
                    attention_weights, tf.zeros_like(attention_weights))
                attention_weights = attention_weights / tf.reduce_sum(attention_weights, axis=1, keep_dims=True)  # B x M
                product = lstm_outputs * attention_weights[:, :, numpy.newaxis]  # B x M x H
                processed_tensor = tf.reduce_sum(product, axis=1)  # B x H
        else:
            processed_tensor = lstm_output
            self.attention_weights_unnormalised = tf.zeros_like(self.word_ids, dtype=tf.float32)

        if self.config["hidden_layer_size"] > 0:
            processed_tensor = tf.layers.dense(
                inputs=processed_tensor, units=self.config["hidden_layer_size"],
                activation=tf.tanh, kernel_initializer=self.initializer)    # B x hidden_layer_size

        self.sentence_scores = tf.layers.dense(
            inputs=processed_tensor, units=1,
            activation=tf.sigmoid, kernel_initializer=self.initializer, name="output_ff")  # [B x 1]
        self.sentence_scores = tf.reshape(self.sentence_scores, shape=[tf.shape(processed_tensor)[0]])  # [B]

        self.loss += self.config["sentence_objective_weight"] * tf.reduce_sum(
            self.sentence_objective_weights * tf.square(self.sentence_scores - self.sentence_labels))

        # Loss for attention, L_attn (in the paper)
        if self.config["attention_objective_weight"] > 0.0:
            self.loss += self.config["attention_objective_weight"] * (
                tf.reduce_sum(
                    self.sentence_objective_weights * tf.square(
                        tf.reduce_max(
                            tf.where(
                                tf.sequence_mask(self.sentence_lengths),
                                self.attention_weights_unnormalised,
                                tf.zeros_like(self.attention_weights_unnormalised) - 1e6),
                            axis=-1) - self.sentence_labels))
                +
                tf.reduce_sum(
                    self.sentence_objective_weights * tf.square(
                        tf.reduce_min(
                            tf.where(
                                tf.sequence_mask(self.sentence_lengths),
                                self.attention_weights_unnormalised,
                                tf.zeros_like(self.attention_weights_unnormalised) + 1e6),
                            axis=-1) - 0.0)))

        self.token_scores = [tf.where(tf.sequence_mask(self.sentence_lengths),
                                      self.attention_weights_unnormalised,
                                      tf.zeros_like(self.attention_weights_unnormalised) - 1e6)]
        self.multihead_token_scores = self.token_scores
        self.multihead_sentence_scores = self.sentence_scores
        """

        # This is the token-level language modelling objective, L_LM
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
            learningrate=self.learningrate,
            clip=self.config["clip"])
        print("Notwork built.")

    def my_multihead_attention(self, inputs, num_heads, reuse=False):
        with tf.variable_scope("multihead_attention", reuse=reuse):
            # Input has shape [B x M x E]

            # Define the number of units = C (adjust it slightly so that it divides the number of heads exactly).
            num_units = math.ceil(inputs.get_shape().as_list()[-1] / num_heads) * num_heads

            # Project the inputs onto a linear space, shape remains [B x M x E]
            keys = tf.layers.dense(inputs, units=num_units, activation=None)

            # Duplicate the input for the appropriate number of heads. Shape is [B x M x (E * num_heads)]
            # keys = tf.tile(keys, (1, 1, num_heads))

            # Split and concatenate. Shape is [(B * num_heads) x M x (E / num_heads)]
            keys = tf.concat(tf.split(keys, num_or_size_splits=num_heads, axis=2), axis=0)

            # Pass inputs through attention layer. Shape is [(B * num_heads) x M x Attention_size]
            attention_evidence = tf.layers.dense(
                inputs=keys, units=self.config["attention_evidence_size"],
                activation=tf.tanh, kernel_initializer=self.initializer)

            # Shape is [(B * num_heads) x M x 1]
            unnormalized_attention_weights = tf.layers.dense(
                inputs=attention_evidence, units=1,
                activation=None, kernel_initializer=self.initializer)

            # Shape is [(B * num_heads) x M]
            unnormalized_attention_weights = tf.squeeze(unnormalized_attention_weights, axis=-1)

            # Unnormalized attention weights, shape is [(B * num_heads) x M].
            unnormalized_attention_weights = tf.sigmoid(unnormalized_attention_weights)

            # Normalized attention weights
            multiply = tf.constant([num_heads])  # make it [B * num_heads]
            this_seq_lengths = tf.tile(self.sentence_lengths, multiply)
            attention_weights = tf.where(
                tf.sequence_mask(this_seq_lengths),
                unnormalized_attention_weights,
                tf.zeros_like(unnormalized_attention_weights))
            attention_weights = attention_weights / tf.reduce_sum(attention_weights, axis=1, keep_dims=True)

            # Get sentence scores, multiply input by attention weights
            product = keys * attention_weights[:, :, numpy.newaxis]  # [(B * num_heads) x M x (E / num_heads)]
            processed_tensor = tf.reduce_sum(product, axis=1)  # [(B * num_heads) x (E / num_heads)]

            # Pass processed tensors through dense layer and obtain sentence scores
            processed_tensor = tf.layers.dense(
                inputs=processed_tensor, units=self.config["hidden_layer_size"],
                activation=tf.tanh, kernel_initializer=self.initializer)  # [(B * num_heads) x hidden_layer_size]

            sentence_scores = tf.layers.dense(
                inputs=processed_tensor, units=1,
                activation=tf.sigmoid, kernel_initializer=self.initializer, name="ff_multi")  # [(B * num_heads) x 1]
            sentence_scores = tf.reshape(sentence_scores, shape=[tf.shape(processed_tensor)[0]])  # [(B * num_heads)]

            # Obtain token scores for each attention head in batch of sentences
            final_token_scores = tf.expand_dims(
                unnormalized_attention_weights, axis=-1)  # [(B * num_heads) x M x 1]
            final_token_scores = tf.concat(
                tf.split(final_token_scores, num_or_size_splits=num_heads, axis=0), axis=2)  # [B x M x num_heads]

            # Obtain sentence scores for each attention head in batch of sentences
            final_sentence_scores = tf.expand_dims(
                sentence_scores, axis=-1)  # [(B * num_heads) x 1]
            final_sentence_scores = tf.concat(
                tf.split(final_sentence_scores, num_or_size_splits=num_heads, axis=0), axis=1)   # [B x num_heads]

            return final_token_scores, final_sentence_scores

    def normalize(self, input, epsilon=1e-8):
        with tf.variable_scope("norm"):
            inputs_shape = input.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(input, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (input - mean) / ((variance + epsilon) ** 0.5)
            outputs = tf.add(tf.multiply(gamma, normalized), beta)
        return outputs

    def multihead_attention(self, queries, keys, num_heads=2,
                            add_residual_connection=False, reuse=False):
        with tf.variable_scope("multihead-attention_v2", reuse=reuse):
            # Current shape of queries, keys and values is:
            # [B x M_q x C], [B x M_k x C] and [B x M_k x C], respectively.

            # Transform everything to shape: [B x M_ x (C * num_heads)]
            # queries = tf.tile(queries, (1, 1, num_heads))
            # keys = tf.tile(keys, (1, 1, num_heads))
            # values = tf.tile(keys, (1, 1, num_heads))

            # Define the number of units
            num_units = queries.get_shape().as_list()[-1]

            # Linearly project Q, K, and V with different linear projections.
            # If copied each embedding, shape is [B x M_ x (C * num_heads)], else [B x M_ x C]
            Q = tf.layers.dense(queries, units=num_units, activation=None)
            K = tf.layers.dense(keys, units=num_units, activation=None)
            V = tf.layers.dense(keys, units=num_units, activation=None)

            # If copied, shape is: [(B * num_heads) x M_ x C],
            # else shape is:       [(B * num_heads) x M_ x C / num_heads]
            Q = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
            K = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
            V = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

            # Dot-Product attention between queries and keys.
            # Shape is: [(B * num_heads) x M_q x M_k].
            attention = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))

            # Scale the attention, shape remains [(B * num_heads) x M_q x M_k].
            model_size = K.get_shape().as_list()[-1]
            attention = tf.div(attention, tf.sqrt(tf.to_float(model_size)))

            # Apply softmax, shape remains [(B * num_heads) x M_q x M_k].
            attention = tf.nn.softmax(attention)

            # Multiply by values.
            # Shape is [(B * num_heads) x M_k x C] if duplicated, else [(B * num_heads) x M_k x C / num_heads].
            outputs = tf.matmul(attention, V)

            # Concat block. Multi-head attentions.
            # Shape is [B x M_k x (C * num_heads)] if duplicated, else [B x M_k x C].
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

            # Linear block. Add a feed-forward layer.
            # Shape is [B x M_k x (C * num_heads)] if duplicated, else [B x M_k x C].
            output = tf.layers.dense(outputs, units=num_units, activation=None)

            # Process attentions (NOT from the paper, but from my head, not sure if fine...)
            attentions = tf.expand_dims(attention, -1)
            attentions = tf.concat(tf.split(attentions, num_heads, axis=0), axis=3)
            attentions = tf.layers.dense(attentions, units=num_heads, activation=tf.nn.softmax)

            if add_residual_connection:
                output = output + queries
                output = self.normalize(output)

        return output, attentions

    def scaled_dot_product_attention(self, keys, values, model_size=None):
        if model_size is None:
            model_size = keys.get_shape().as_list()[-1]

        # keys_times_values = tf.div(tf.matmul(a=queries, b=tf.transpose(keys, [0, 2, 1])),
        #                            tf.sqrt(model_size))  # (B * num_heads) x M x M
        # attentions = tf.nn.softmax(keys_times_values)  # (B * num_heads) x M x M
        # scaled_dprod_att = tf.matmul(attentions, values)

        attentions = tf.layers.dense(keys, units=200, activation=tf.nn.sigmoid, kernel_initializer=self.initializer)
        # attentions = tf.layers.dense(attentions, units=1, activation=None)
        attentions = tf.nn.sigmoid(attentions)  # (B * num_heads) x M x 1
        attentions = attentions / tf.reduce_sum(attentions, axis=1, keep_dims=True)  # B x M
        product = values * attentions
        self.keys_times_values = tf.reduce_sum(product, axis=1)
        scaled_dprod_att = product
        # scaled_dprod_att = tf.matmul(attentions, values)
        return scaled_dprod_att, attentions

    def construct_lmcost(self, input_tensor_fw, input_tensor_bw,
                         sentence_lengths, target_ids, lmcost_type, name):
        with tf.variable_scope(name):
            lmcost_max_vocab_size = min(len(self.word2id), self.config["lmcost_max_vocab_size"])
            target_ids = tf.where(
                tf.greater_equal(target_ids, lmcost_max_vocab_size - 1),
                x=(lmcost_max_vocab_size - 1) + tf.zeros_like(target_ids),
                y=target_ids)
            cost = 0.0
            if lmcost_type == "separate":
                lmcost_fw_mask = tf.sequence_mask(sentence_lengths, maxlen=tf.shape(target_ids)[1])[:, 1:]
                lmcost_bw_mask = tf.sequence_mask(sentence_lengths, maxlen=tf.shape(target_ids)[1])[:, :-1]
                lmcost_fw = self._construct_lmcost(
                    input_tensor_fw[:, :-1, :],
                    lmcost_max_vocab_size,
                    lmcost_fw_mask,
                    target_ids[:, 1:],
                    name=name+"_fw")
                lmcost_bw = self._construct_lmcost(
                    input_tensor_bw[:, 1:, :],
                    lmcost_max_vocab_size,
                    lmcost_bw_mask,
                    target_ids[:, :-1],
                    name=name+"_bw")
                cost += lmcost_fw + lmcost_bw
            elif lmcost_type == "joint":
                joint_input_tensor = tf.concat(
                    [input_tensor_fw[:, :-2, :], input_tensor_bw[:, 2:, :]], axis=-1)
                lmcost_mask = tf.sequence_mask(sentence_lengths, maxlen=tf.shape(target_ids)[1])[:, 1:-1]
                cost += self._construct_lmcost(
                    joint_input_tensor,
                    lmcost_max_vocab_size,
                    lmcost_mask,
                    target_ids[:, 1:-1],
                    name=name+"_joint")
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

    def construct_optimizer(self, opt_strategy, loss, learningrate, clip):
        if opt_strategy == "adadelta":
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learningrate)
        elif opt_strategy == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learningrate)
        elif opt_strategy == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningrate)
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

    def translate2id(self, token, token2id, unk_token=None,
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

    def create_input_dictionary_for_batch(self, batch, is_training, learningrate):
        if self.config["binary_labels"]:
            sentence_lengths = numpy.array([len(sentence) for sentence in batch])
            max_sentence_length = sentence_lengths.max()
            max_word_length = numpy.array(
                [numpy.array([len(word[0]) for word in sentence]).max() for sentence in batch]).max()

            if 0 < self.config["allowed_word_length"] < max_word_length:
                max_word_length = min(max_word_length, self.config["allowed_word_length"])

            word_ids = numpy.zeros((len(batch), max_sentence_length), dtype=numpy.int32)
            char_ids = numpy.zeros((len(batch), max_sentence_length, max_word_length), dtype=numpy.int32)
            word_lengths = numpy.zeros((len(batch), max_sentence_length), dtype=numpy.int32)
            word_labels = numpy.zeros((len(batch), max_sentence_length), dtype=numpy.float32)
            sentence_labels = numpy.zeros((len(batch)), dtype=numpy.float32)
            word_objective_weights = numpy.zeros((len(batch), max_sentence_length), dtype=numpy.float32)
            sentence_objective_weights = numpy.zeros((len(batch)), dtype=numpy.float32)

            # A proportion of the sentences should be assigned to UNK token embedding. Do this just for training.
            singletons = self.singletons if is_training else None
            singletons_prob = self.config["singletons_prob"] if is_training else 0.0

            # Go through all the sentences, words and characters in the batch
            for i in range(len(batch)):
                count_interesting_labels = numpy.array(
                    [1.0 if batch[i][j][-1] != self.config["default_label"]
                     else 0.0 for j in range(len(batch[i]))]).sum()
                sentence_labels[i] = 1.0 if count_interesting_labels >= 1.0 else 0.0

                for j in range(len(batch[i])):
                    # Translate words to ids
                    word_ids[i][j] = self.translate2id(
                        token=batch[i][j][0],
                        token2id=self.word2id,
                        unk_token=self.UNK,
                        lowercase=self.config["lowercase"],
                        replace_digits=self.config["replace_digits"],
                        singletons=singletons,
                        singletons_prob=singletons_prob)
                    word_labels[i][j] = 0.0 if batch[i][j][-1] == self.config["default_label"] else 1.0
                    word_lengths[i][j] = len(batch[i][j][0])
                    for k in range(min(len(batch[i][j][0]), max_word_length)):
                        # Translate chars to ids, never lowercase or replace digits, no singleton chars or probs.
                        char_ids[i][j][k] = self.translate2id(
                            token=batch[i][j][0][k],
                            token2id=self.char2id,
                            unk_token=self.CUNK)

                    # Here: What does this mean?
                    if len(batch[i][j]) == 2 or (len(batch[i][j]) >= 3 and batch[i][j][1] == "T"):
                        word_objective_weights[i][j] = 1.0
                if len(batch[i][j]) == 2 or (len(batch[i][j]) >= 3 and batch[i][0][1] == "S") \
                        or self.config["sentence_objective_persistent"]:
                    sentence_objective_weights[i] = 1.0
                # print("Words: ", [batch[i][k][0] for k in range(len(batch[i]) - 1)])
                # print("Sentence label: ", sentence_labels[i])
                # print("Word labels: ", word_labels[i])
        else:
            sentence_lengths = numpy.array([len(sentence[:-1]) for sentence in batch])
            max_sentence_length = sentence_lengths.max()
            max_word_length = numpy.array(
                [numpy.array([len(word[0]) for word in sentence[:-1]]).max() for sentence in batch]).max()

            if 0 < self.config["allowed_word_length"] < max_word_length:
                max_word_length = min(max_word_length, self.config["allowed_word_length"])

            word_ids = numpy.zeros((len(batch), max_sentence_length), dtype=numpy.int32)
            char_ids = numpy.zeros((len(batch), max_sentence_length, max_word_length), dtype=numpy.int32)
            word_lengths = numpy.zeros((len(batch), max_sentence_length), dtype=numpy.int32)
            word_labels = numpy.zeros((len(batch), max_sentence_length), dtype=numpy.float32)
            sentence_labels = numpy.zeros((len(batch)), dtype=numpy.float32)
            word_objective_weights = numpy.zeros((len(batch), max_sentence_length), dtype=numpy.float32)
            sentence_objective_weights = numpy.zeros((len(batch)), dtype=numpy.float32)

            # A proportion of the sentences should be assigned to UNK token embedding. Do this just for training.
            singletons = self.singletons if is_training else None
            singletons_prob = self.config["singletons_prob"] if is_training else 0.0

            # Go through all the sentences, words and characters in the batch
            for i in range(len(batch)):
                sentence_labels[i] = self.sent_label2id[batch[i][-1][-1]]

                # Skip the last word-pair in the sentence as it contains just the sentence label
                for j in range(len(batch[i]) - 1):
                    # Translate words to ids
                    word_ids[i][j] = self.translate2id(
                        token=batch[i][j][0],
                        token2id=self.word2id,
                        unk_token=self.UNK,
                        lowercase=self.config["lowercase"],
                        replace_digits=self.config["replace_digits"],
                        singletons=singletons,
                        singletons_prob=singletons_prob)
                    word_labels[i][j] = self.token_label2id[batch[i][j][-1]]
                    word_lengths[i][j] = len(batch[i][j][0])
                    for k in range(min(len(batch[i][j][0]), max_word_length)):
                        # Translate chars to ids, never lowercase or replace digits, no singleton chars or probs.
                        char_ids[i][j][k] = self.translate2id(
                            token=batch[i][j][0][k],
                            token2id=self.char2id,
                            unk_token=self.CUNK)

                    # Here: What does this mean?
                    if len(batch[i][j]) == 2 or (len(batch[i][j]) >= 3 and batch[i][j][1] == "T"):
                        word_objective_weights[i][j] = 1.0
                if len(batch[i][j]) == 2 or (len(batch[i][j]) >= 3 and batch[i][0][1] == "S") \
                        or self.config["sentence_objective_persistent"]:
                    sentence_objective_weights[i] = 1.0
                # print("Max sentence length: ", max_sentence_length)
                # print("Word lengths: ", [word_lengths[i][rr] for rr in range(len(batch[i]) - 1)])
                # print("Words: ", [batch[i][rrr][0] for rrr in range(len(batch[i]) - 1)])
                # print("Sentence label: ", sentence_labels[i])
                # print("Word labels: ", word_labels[i])
                # print("==================================")

        if self.config["debug_mode"]:
            with open("my_output.txt", "a") as f:
                f.write("\n================================================\n\n")
                f.write("Word ids shape: " + str(word_ids.shape) + "\n")
                f.write("Char ids shape: " + str(char_ids.shape) + "\n")
                f.write("Sentence objective weights: " + str(sentence_objective_weights.shape) + "\n")
                f.write("Word objective weights: " + str(word_objective_weights.shape) + "\n")
                f.write("Word labels:\n")
                for row in word_labels.tolist():
                    f.write(" ".join([str(r) for r in row]))
                    f.write("\n")
                f.write("Sentence labels:\n")
                f.write(" ".join([str(s) for s in sentence_labels.tolist()]) + "\n")
                f.write("Sentence lengths: " + " ".join([str(s) for s in sentence_lengths]))
                f.write("\nWord lengths:\n")
                for row in word_lengths.tolist():
                    f.write(" ".join([str(r) for r in row]))
                    f.write("\n")

        input_dictionary = {
            self.word_ids: word_ids,
            self.char_ids: char_ids,
            self.sentence_lengths: sentence_lengths,
            self.word_lengths: word_lengths,
            self.word_labels: word_labels,
            self.word_objective_weights: word_objective_weights,
            self.sentence_labels: sentence_labels,
            self.sentence_objective_weights: sentence_objective_weights,
            self.learningrate: learningrate,
            self.is_training: is_training}
        return input_dictionary

    def process_batch(self, batch, is_training, learningrate):
        feed_dict = self.create_input_dictionary_for_batch(batch, is_training, learningrate)
        cost, sentence_scores, token_scores, multihead_sentence_scores, multihead_token_scores, mm = self.session.run(
            [self.loss, self.sentence_scores, self.token_scores,
             self.multihead_sentence_scores, self.multihead_token_scores, self.mm] +
            ([self.train_op] if is_training else []), feed_dict=feed_dict)[:6]
        if self.config["debug_mode"]:
            with open("my_output.txt", "a") as f:
                f.write("\n\nCOST = " + str(cost) + "\n")
                f.write("Sentence scores of shape " + str(sentence_scores.shape) + " \n")
                f.write("Multihead_sentence_scores of shape " + str(multihead_sentence_scores.shape) + " \n")
                f.write("Tokens scores LENGTH "
                        + str(len(token_scores)) + " x "
                        + str(len(token_scores[0])) + " x "
                        + str(len(token_scores[0][0])) + " \n")
                f.write("Multihead_token_scores LENGTH "
                        + str(len(multihead_token_scores)) + " x "
                        + str(len(multihead_token_scores[0])) + " x "
                        + str(len(multihead_token_scores[0][0])) + " \n")
                f.write("Sentence scores:\n" + " ".join([str(s) for s in sentence_scores]) + "\n")
                f.write("Multihead_sentence_scores:\n" +
                        " ".join([str(sent_score) for sent_score in multihead_sentence_scores]) + "\n")
                f.write("Word scores:\n")
                for token_scores_per_sentence in token_scores:
                    f.write(" ".join([str(token_score_heads) for token_score_heads in token_scores_per_sentence]))
                    f.write("\n")
                f.write("Multihead_token_scores:\n")
                for sentence in multihead_token_scores[0]:
                    f.write(" ".join([str(token) for token in sentence]))
                    f.write("\n")
                f.write("mm = " + str(mm.shape) + "\n" + " ".join([str(m) for m in mm]))
                f.write("\n")
        return cost, multihead_sentence_scores, multihead_token_scores

    def initialize_session(self):
        tf.set_random_seed(self.config["random_seed"])
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = self.config["tf_allow_growth"]
        session_config.gpu_options.per_process_gpu_memory_fraction = self.config[
            "tf_per_process_gpu_memory_fraction"]
        self.session = tf.Session(config=session_config)
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)

    def get_parameter_count(self):
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
        dump = {}
        dump["config"] = self.config
        dump["UNK"] = self.UNK
        dump["CUNK"] = self.CUNK
        dump["word2id"] = self.word2id
        dump["char2id"] = self.char2id
        dump["singletons"] = self.singletons

        dump["params"] = {}
        for variable in tf.global_variables():
            assert(
                variable.name not in dump["params"]),\
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

            labeler = MLTModel(dump["config"])
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
                assert(variable.name in dump["params"]), "Variable not in dump: " + str(variable.name)
                assert(variable.shape == dump["params"][variable.name].shape), \
                    "Variable shape not as expected: " + str(variable.name) \
                    + " " + str(variable.shape) + " " \
                    + str(dump["params"][variable.name].shape)
                value = numpy.asarray(dump["params"][variable.name])
                self.session.run(variable.assign(value))

