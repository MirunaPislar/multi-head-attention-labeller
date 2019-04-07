from math import ceil
import tensorflow as tf
import tensorflow_probability as tfp


def kl_divergence_loss(expected_logits, actual_logits):
    """KL divergence loss"""
    p = tfp.distributions.Categorical(logits=expected_logits)
    q = tfp.distributions.Categorical(logits=actual_logits)
    return tfp.distributions.kl_divergence(p, q)


def combine_attentions(attention_list):
    """Combine different layer attentions and then average over layers/heads."""
    # Stack all hidden layer attention tensors to get a tensor with shape
    # [num_hidden_layers, batch_size, num_heads, target_length, input_length].
    attentions = tf.stack(attention_list)
    # Reduce mean across all layers (axis=0) and all heads (axis=2) to get a
    # tensor with shape [batch_size, target_length, input_length].
    return tf.reduce_mean(attentions, [0, 2])


def mse_loss(expected_logits, actual_weights):
    expected_weights = tf.nn.softmax(expected_logits)
    return tf.losses.mean_squared_error(expected_weights, actual_weights)


def single_head_attention(inputs, initializer, attention_size,
                          sentence_lengths, hidden_units):
    """
    Compute single head attention (just normal, vanilla, soft attention).
    :param inputs: 3D floats of shape [B x M x E]
    :param initializer: type of initializer (best if glorot or xavier)
    :param attention_size: number of units to use for the attention evidence
    :param sentence_lengths: 2D ints of shape [B x M]
    :param hidden_units: number of units to use for the processed vector
    :return sentence_scores: result of the attention * input; floats of shape [B]
    :return sentence_predictions: predicted labels for each sentence in the batch; ints of shape [B]
    :return token_scores: result of the un-normalized attention weights; floats of shape [B x M]
    :return token_predictions: predicted labels for each token in each sentence; ints of shape [B x M]
    """
    with tf.variable_scope("single_head_attention"):
        attention_evidence = tf.layers.dense(
            inputs=inputs, units=attention_size,
            activation=tf.tanh, kernel_initializer=initializer)  # [B x M x attention_size]
        attention_weights = tf.layers.dense(
            inputs=attention_evidence, units=1,
            activation=None, kernel_initializer=initializer)  # [B x M x 1]
        attention_weights = tf.squeeze(attention_weights, axis=-1)  # [B x M]
        attention_weights = tf.sigmoid(attention_weights)

        token_scores = attention_weights
        token_predictions = tf.where(
            tf.greater_equal(token_scores, 0.5),
            tf.ones_like(token_scores),
            tf.zeros_like(token_scores))
        token_predictions = tf.cast(tf.where(
            tf.sequence_mask(sentence_lengths),
            token_predictions,
            tf.zeros_like(token_predictions) - 1e6), tf.int32)

        attention_weights = tf.where(
            tf.sequence_mask(sentence_lengths),
            attention_weights, tf.zeros_like(attention_weights))
        attention_weights = attention_weights / tf.reduce_sum(attention_weights, axis=1,
                                                              keep_dims=True)  # [B x M]
        product = inputs * tf.expand_dims(attention_weights, axis=-1)  # [B x M x E]
        processed_tensor = tf.reduce_sum(product, axis=1)  # [B x E]

        if hidden_units > 0:
            processed_tensor = tf.layers.dense(
                inputs=processed_tensor, units=hidden_units,
                activation=tf.tanh, kernel_initializer=initializer)  # [B x hidden_units]

        sentence_scores = tf.layers.dense(
            inputs=processed_tensor, units=1,
            activation=tf.sigmoid, kernel_initializer=initializer,
            name="output_sent_single_head_ff")  # [B x 1]
        sentence_scores = tf.reshape(
            sentence_scores, shape=[tf.shape(processed_tensor)[0]])  # [B]
        sentence_predictions = tf.where(
            tf.greater_equal(sentence_scores, 0.5),
            tf.ones_like(sentence_scores, dtype=tf.int32),
            tf.zeros_like(sentence_scores, dtype=tf.int32))     # [B]
        return sentence_scores, sentence_predictions, token_scores, token_predictions


def multi_head_attention(inputs, initializer, attention_size,
                         sentence_lengths, hidden_units,
                         num_sentence_labels, num_token_labels):
    """
    Compute single head attention (just normal, vanilla, soft attention).
    :param inputs: 3D floats of shape [B x M x E]
    :param initializer: type of initializer (best if glorot or xavier)
    :param attention_size: number of units to use for the attention evidence
    :param sentence_lengths: 2D ints of shape [B x M]
    :param hidden_units: number of units to use for the processed vector
    :param num_sentence_labels: number of unique sentence labels
    :param num_token_labels: number of unique token labels
    :return sentence_scores: 2D floats of shape [B x num_sentence_labels]
    :return sentence_predictions: predicted labels for each sentence in the batch; ints of shape [B]
    :return token_scores: 3D floats of shape [B x M x num_token_labels]
    :return token_predictions: predicted labels for each token in each sentence; ints of shape [B x M]
    """
    with tf.variable_scope("multi_head_attention"):
        attention_evidence = tf.layers.dense(
            inputs=inputs, units=attention_size,
            activation=tf.tanh, kernel_initializer=initializer)  # [B x M x attention_size]

        token_scores = tf.layers.dense(
            inputs=attention_evidence, units=num_token_labels,
            kernel_initializer=initializer,
            name="output_multi_tokens_scores_ff")  # [B x M x num_token_labels]
        token_proba = tf.nn.softmax(token_scores)
        token_predictions = tf.argmax(token_proba, axis=2, output_type=tf.int32)  # [B x M]

        attention_weights = tf.layers.dense(
            inputs=attention_evidence, units=1,
            kernel_initializer=initializer)  # [B x M x 1]
        attention_weights = tf.squeeze(attention_weights, axis=-1)  # [B x M]
        attention_weights = tf.nn.softmax(attention_weights)
        attention_weights = tf.where(
            tf.sequence_mask(sentence_lengths),
            attention_weights, tf.zeros_like(attention_weights))
        attention_weights = attention_weights / tf.reduce_sum(attention_weights, axis=1,
                                                              keep_dims=True)  # [B x M]
        product = inputs * tf.expand_dims(attention_weights, axis=-1)  # [B x M x E]
        processed_tensor = tf.reduce_sum(product, axis=1)  # [B x E]

        if hidden_units > 0:
            processed_tensor = tf.layers.dense(
                inputs=processed_tensor, units=hidden_units,
                activation=tf.tanh, kernel_initializer=initializer)  # [B x hidden_units]

        sentence_scores = tf.layers.dense(
            inputs=processed_tensor, units=num_sentence_labels,
            kernel_initializer=initializer,
            name="output_multi_sent_specified_scores_ff")  # [B x num_unique_sent_labels]
        sentence_probabilities = tf.nn.softmax(sentence_scores)
        sentence_predictions = tf.argmax(sentence_probabilities, axis=1)  # [B]
        return sentence_scores, sentence_predictions, token_scores, token_predictions


def transformer_attention(inputs, initializer,
                          hidden_units,
                          num_sentence_labels, num_heads,
                          is_training, dropout,
                          use_residual_connection,
                          token_scoring_method):
    """
    Compute the multi-head transformer architecture.
    :param inputs: 3D floats of shape [B x M x E]
    :param initializer: type of initializer (best if glorot or xavier)
    :param hidden_units: number of units to use for the processed vector
    :param num_sentence_labels: number of unique sentence labels
    :param num_heads: number of unique token labels
    :param is_training: if set to True, the current phase is a training one (rather than testing)
    :param dropout: the keep_probs value for the dropout
    :param use_residual_connection: if set to True, a residual connection is added to the inputs
    :param token_scoring_method: can be either max, sum or avg
    :return sentence_scores: 2D floats of shape [B x num_sentence_labels]
    :return sentence_predictions: predicted labels for each sentence in the batch; ints of shape [B]
    :return token_scores: 3D floats of shape [B x M x num_heads]
    :return token_predictions: predicted labels for each token in each sentence; ints of shape [B x M]
    """
    with tf.variable_scope("transformer_multi_head_attention"):
        num_units = ceil(inputs.get_shape().as_list()[-1] / num_heads) * num_heads

        # Linear projection of the input (used to compute the attention values).
        adapted_inputs = tf.layers.dense(inputs, num_units)  # [B x M x num_units]

        # Project to get the queries, keys, and values.
        queries = tf.layers.dense(
            inputs, num_units, activation=tf.tanh,
            kernel_initializer=initializer)  # [B x M x num_units]
        keys = tf.layers.dense(
            inputs, num_units, activation=tf.tanh,
            kernel_initializer=initializer)  # [B x M x num_units]
        values = tf.layers.dense(
            inputs, num_units, activation=tf.tanh,
            kernel_initializer=initializer)  # [B x M x num_units]

        # Split and concat
        queries = tf.concat(tf.split(queries, num_heads, axis=2), axis=0)  # [(heads * B) x M x (num_units / heads)]
        keys = tf.concat(tf.split(keys, num_heads, axis=2), axis=0)  # [(heads * B) x M x (num_units / heads)]
        values = tf.concat(tf.split(values, num_heads, axis=2), axis=0)  # [(heads * B) x M x (num_units / heads)]

        # Transpose multiplication and scale
        attention_evidence = tf.matmul(queries, tf.transpose(keys, [0, 2, 1]))  # [(heads * B) x M x M]
        attention_evidence = tf.math.divide(attention_evidence, tf.constant(num_units ** 0.5))

        # Add a softmax layer
        attention_weights = tf.nn.softmax(attention_evidence)  # [(heads * B) x M x M]

        # Dropouts
        dropout_attention = (dropout * tf.cast(is_training, tf.float32)
                             + (1.0 - tf.cast(is_training, tf.float32)))

        attention_weights = tf.nn.dropout(
            attention_weights, dropout_attention,
            name="dropout_transducer_attention_weights")    # [(heads * B) x M x M]

        product = tf.matmul(attention_weights, values)  # [(heads * B) x M x (num_units / heads)]
        product = tf.concat(tf.split(product, num_heads, axis=0), axis=2)  # [B x M x num_units]

        # Add a residual connection, followed by layer normalization.
        if use_residual_connection:
            product += adapted_inputs
            product = layer_normalization(product)  # [B x M x num_units]
        processed_tensor = tf.reduce_sum(product, axis=1)  # [B x num_units]

        if hidden_units > 0:
            processed_tensor = tf.layers.dense(
                inputs=processed_tensor, units=hidden_units,
                activation=tf.tanh, kernel_initializer=initializer)  # [B x hidden_units]

        sentence_scores = tf.layers.dense(
            inputs=processed_tensor, units=num_sentence_labels,
            kernel_initializer=initializer,
            name="output_sent_specified_scores_ff")  # [B x num_unique_sent_labels]
        sentence_probabilities = tf.nn.softmax(sentence_scores)
        sentence_predictions = tf.argmax(sentence_probabilities, axis=1)  # [B]

        if token_scoring_method == "sum":
            token_scores = tf.expand_dims(tf.reduce_sum(attention_evidence, axis=1), axis=2)  # [B x M x 1]
        elif token_scoring_method == "max":
            token_scores = tf.expand_dims(tf.reduce_max(attention_evidence, axis=1), axis=2)  # [B x M x 1]
        else:
            token_scores = tf.expand_dims(tf.reduce_mean(attention_evidence, axis=1), axis=2)  # [B x M x 1]
        token_scores = tf.concat(tf.split(token_scores, num_heads, axis=0), axis=2)  # [B x M x num_heads]
        token_proba = tf.nn.softmax(token_scores)
        token_predictions = tf.argmax(token_proba, axis=2, output_type=tf.int32)  # [B x M]

        return sentence_scores, sentence_predictions, token_scores, token_predictions, token_proba


def layer_normalization(layer, epsilon=1e-8):
    """
    Implementation of layer normalization.
    :param layer: has at least 2D shape, with the first one being batch_size
    :param epsilon: a small number to avoid numerical issues, such as zero division.
    :return: normalized tensor, of the same shape as the input
    """
    with tf.variable_scope("layer_norm"):
        params_shape = layer.get_shape()[-1:]
        mean, variance = tf.nn.moments(layer, [-1], keep_dims=True)
        beta = tf.get_variable(
            name="beta", shape=params_shape, initializer=tf.zeros_initializer(), trainable=True)
        gamma = tf.get_variable(
            name="gamma", shape=params_shape, initializer=tf.ones_initializer(), trainable=True)
        normalized = (layer - mean) / ((variance + epsilon) ** 0.5)
        outputs = gamma * normalized + beta
    return outputs


def division_masking(inputs, axis, multiplies):
    """
    Masking used when dividing one element by the sum on a certain axis.
    Division by 0 is not possible -- all values will be -infinity, instead.
    :param inputs: the input needed to be divided
    :param axis: axis on which to perform the reduced sum
    :param multiplies: the shape to be used when tiling the division masks.
    :return: the correct normalized inputs (with -infinity for divisions by 0).
    """
    division_masks = tf.sign(tf.reduce_sum(inputs, axis=axis, keep_dims=True))
    division_masks = tf.tile(division_masks, multiples=multiplies)
    divided_inputs = tf.where(
        tf.equal(division_masks, 0),
        tf.zeros_like(inputs),
        # tf.ones_like(inputs) * (-2 ** 32 + 1.0),
        tf.div(inputs, tf.reduce_sum(inputs, axis=axis, keep_dims=True)))
    return divided_inputs


def transformer_attention_version2(
        inputs,
        initializer,
        attention_activation,
        num_sentence_labels,
        num_heads,
        is_training,
        dropout,
        sentence_lengths,
        normalize_sentence,
        token_scoring_method,
        separate_heads_for_sentence_scores=True):
    """
    Compute the multi-head transformer architecture.
    :param inputs: 3D floats of shape [B x M x E]
    :param initializer: type of initializer (best if glorot or xavier)
    :param attention_activation: type of attention activation (linear, softmax or sigmoid)
    :param num_sentence_labels: number of unique sentence labels
    :param num_heads: number of unique token labels
    :param is_training: if set to True, the current phase is a training one (rather than testing)
    :param dropout: the keep_probs value for the dropout
    :param sentence_lengths: the true sentence lengths, used for masking
    :param normalize_sentence: if set to True, the last weighted sentence layer is normalized
    :param token_scoring_method: can be either max, sum or avg
    :param separate_heads_for_sentence_scores: boolean value; when set to False, all heads
    are used to obtain the sentence scores; when set to True, the default and non-default heads
    from the token scores are used to obtain the sentence scores.
    :return sentence_scores: 2D floats of shape [B x num_sentence_labels]
    :return sentence_predictions: predicted labels for each sentence in the batch; ints of shape [B]
    :return token_scores: 3D floats of shape [B x M x num_heads]
    :return token_predictions: predicted labels for each token in each sentence; ints of shape [B x M]
    """
    with tf.variable_scope("transformer_multi_head_attention"):
        num_units = inputs.get_shape().as_list()[-1]
        if num_units % num_heads != 0:
            num_units = ceil(num_units / num_heads) * num_heads
            inputs = tf.layers.dense(inputs, num_units)  # [B x M x num_units]

        # Project to get the queries, keys, and values, all of them of shape [B, M, num_units].
        queries = tf.layers.dense(
            inputs, num_units, activation=tf.tanh,
            kernel_initializer=initializer)
        keys = tf.layers.dense(
            inputs, num_units, activation=tf.tanh,
            kernel_initializer=initializer)
        values = tf.layers.dense(
            inputs, num_units, activation=tf.tanh,
            kernel_initializer=initializer)

        # Split and concat to get shapes [(num_heads * B), M, (num_units / num_heads)].
        queries = tf.concat(tf.split(queries, num_heads, axis=2), axis=0)
        keys = tf.concat(tf.split(keys, num_heads, axis=2), axis=0)

        # Scaled dot-product attention.
        attention_evidence = tf.matmul(queries, tf.transpose(keys, [0, 2, 1]))  # [(num_heads * B), M, M]
        attention_evidence = tf.math.divide(attention_evidence, tf.constant(num_units ** 0.5))

        # Mask out the 0 values in the attention_evidence before proceeding to the non-linear activation.
        attention_evidence = tf.where(
            tf.equal(attention_evidence, 0),
            tf.ones_like(attention_evidence) * (-2 ** 32 + 1),
            attention_evidence)  # [num_heads * B, M, M]

        # Obtain the un-normalized attention weights.
        if attention_activation == "soft":
            attention_weights = tf.nn.sigmoid(attention_evidence)
        elif attention_activation == "sharp":
            attention_weights = tf.exp(attention_evidence)
        elif attention_activation == "linear":
            attention_weights = attention_evidence
        else:
            raise ValueError("Unknown/unsupported activation for attention: %s"
                             % attention_activation)

        # Dropout over the attention weights.
        dropout_attention = (dropout * tf.cast(is_training, tf.float32)
                             + (1.0 - tf.cast(is_training, tf.float32)))

        attention_weights = tf.nn.dropout(
            attention_weights, dropout_attention,
            name="dropout_transformer_attention")

        # attention_mask = tf.tile(
        #     input=tf.expand_dims(tf.sequence_mask(sentence_lengths), axis=-1),
        #     multiples=[1, 1, num_heads * tf.shape(keys)[1]])  # [B x M x num_heads * M]
        # attention_mask = tf.concat(tf.split(attention_mask, num_heads, axis=2), axis=0)  # [(num_heads * B), M, M]

        # attention_weights = tf.where(
        #     attention_mask,
        #     attention_weights_unnormalized,
        #     tf.zeros_like(attention_weights_unnormalized))
        # attention_weights = attention_weights / tf.reduce_sum(attention_weights, axis=1, keep_dims=True)

        # Obtain the token scores from the attention weights.
        # The token_scores below will have shape [(num_heads * B), 1, M].
        if token_scoring_method == "sum":
            token_scores = tf.reduce_sum(attention_weights, axis=1, keep_dims=True)
        elif token_scoring_method == "max":
            token_scores = tf.reduce_max(attention_weights, axis=1, keep_dims=True)
        elif token_scoring_method == "avg":
            token_scores = tf.reduce_mean(attention_weights, axis=1, keep_dims=True)
        else:
            raise ValueError("Unknown/unsupported token scoring method: %s" % token_scoring_method)

        token_scores = tf.concat(tf.split(token_scores, num_heads), axis=1)  # [B, num_heads, M]

        # Mask the token scores
        token_scores_mask = tf.tile(
            input=tf.expand_dims(tf.sequence_mask(sentence_lengths), axis=1),
            multiples=[1, num_heads, 1])  # [B, num_heads, M]

        token_scores = tf.where(token_scores_mask, token_scores, tf.zeros_like(token_scores))

        token_probabilities = division_masking(
            inputs=token_scores, axis=1, multiplies=[1, num_heads, 1])  # [B, num_heads, M]

        token_predictions = tf.argmax(token_probabilities, axis=1, output_type=tf.int32)  # [B, M]

        # Obtain the sentence scores as a weighted sum between the inputs and the attention weights.
        weighted_sum_representation = tf.matmul(token_probabilities, values)  # [B, num_heads, num_units]
        m = weighted_sum_representation

        if normalize_sentence:
            weighted_sum_representation = layer_normalization(weighted_sum_representation)

        processed_tensor = tf.reduce_sum(weighted_sum_representation, axis=2)  # [B, num_heads]
        n = processed_tensor

        if separate_heads_for_sentence_scores:
            sentence_default_score = tf.expand_dims(processed_tensor[:, 0], axis=-1)
            sentence_non_default_scores = tf.layers.dense(
                inputs=processed_tensor[:, 1:], units=num_sentence_labels-1,
                activation=None, kernel_initializer=initializer,
                name="ff_sentence_default_scores")
            sentence_scores = tf.concat([sentence_default_score, sentence_non_default_scores],
                                        axis=-1, name="sentence_scores_concatenation")
        else:
            sentence_scores = tf.layers.dense(
                inputs=processed_tensor, units=num_sentence_labels,
                activation=None, kernel_initializer=initializer,
                name="ff_sentence_scores")  # [B, num_unique_sent_labels]
        p = sentence_scores
        sentence_probabilities = tf.nn.softmax(sentence_scores)
        sentence_predictions = tf.argmax(sentence_probabilities, axis=1)  # [B]
        return sentence_scores, sentence_predictions, tf.transpose(token_scores, [0, 2, 1]), token_predictions, m, n, p

