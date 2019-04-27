from math import ceil
import tensorflow as tf


def layer_normalization(layer, epsilon=1e-8):
    """
    Implementation for layer normalization.
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


def label_smoothing(labels, epsilon=0.1):
    """
    Implementation for label smoothing. This discourages the model to become
    too confident about its predictions and thus, less prone to overfitting.
    Label smoothing regularizes the model and makes it more adaptable.
    :param labels: 3D tensor with the last dimension as the number of labels
    :param epsilon: smoothing rate
    :return: smoothed labels
    """
    num_labels = labels.get_shape().as_list()[-1]
    return ((1 - epsilon) * labels) + (epsilon / num_labels)


def mask(inputs, queries=None, keys=None, type=None):
    """Masks paddings on keys or queries to inputs
    inputs: 3d tensor. (N, T_q, T_k)
    queries: 3d tensor. (N, T_q, d)
    keys: 3d tensor. (N, T_k, d)

    e.g.,
    >> queries = tf.constant([[[1.],
                        [2.],
                        [0.]]], tf.float32) # (1, 3, 1)
    >> keys = tf.constant([[[4.],
                     [0.]]], tf.float32)  # (1, 2, 1)
    >> inputs = tf.constant([[[4., 0.],
                               [8., 0.],
                               [0., 0.]]], tf.float32)
    >> mask(inputs, queries, keys, "key")
    array([[[ 4.0000000e+00, -4.2949673e+09],
        [ 8.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09]]], dtype=float32)
    >> inputs = tf.constant([[[1., 0.],
                             [1., 0.],
                              [1., 0.]]], tf.float32)
    >> mask(inputs, queries, keys, "query")
    array([[[1., 0.],
        [1., 0.],
        [0., 0.]]], dtype=float32)
    """
    padding_num = -2 ** 32 + 1
    if type in ("k", "key", "keys"):
        # Generate masks
        masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
        masks = tf.expand_dims(masks, 1) # (N, 1, T_k)
        masks = tf.tile(masks, [1, tf.shape(queries)[1], 1])  # (N, T_q, T_k)

        # Apply masks to inputs
        paddings = tf.ones_like(inputs) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)  # (N, T_q, T_k)
    elif type in ("q", "query", "queries"):
        # Generate masks
        masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
        masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
        masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)

        # Apply masks to inputs
        outputs = inputs * masks
    elif type in ("f", "future", "right"):
        diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

        paddings = tf.ones_like(masks) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
    else:
        print("Check if you entered type correctly!")
    return outputs


def single_head_attention_binary_labels(
        inputs,
        initializer,
        attention_size,
        sentence_lengths,
        hidden_units):
    """
    Compute single-head attention (just normal, vanilla, soft attention).
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
            kernel_initializer=initializer)  # [B x M x 1]
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


def single_head_attention_multiple_labels(
        inputs,
        initializer,
        attention_size,
        sentence_lengths,
        hidden_units,
        num_sentence_labels,
        num_token_labels):
    """
    Compute single-head attention, but adapt it (naively) to make it work for multiple labels.
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
        token_probabilities = tf.nn.softmax(token_scores)
        token_predictions = tf.argmax(token_probabilities, axis=2, output_type=tf.int32)  # [B x M]

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
        return sentence_scores, sentence_predictions, token_scores, token_predictions, token_probabilities


def multi_head_attention_with_scores_from_shared_heads(
        inputs,
        initializer,
        attention_activation,
        hidden_units,
        num_sentence_labels,
        num_heads,
        is_training,
        dropout,
        sentence_lengths,
        use_residual_connection,
        token_scoring_method):
    """
    Compute multi-head attention (mainly inspired by the transformer architecture).
    This method does not take into account any masking at any level.
    All the masking will be performed before computing a primary/secondary loss.
    :param inputs: 3D floats of shape [B x M x E]
    :param initializer: type of initializer (best if glorot or xavier)
    :param attention_activation: type of attention activation (linear, softmax or sigmoid)
    :param hidden_units: number of units to use for the processed vector
    :param num_sentence_labels: number of unique sentence labels
    :param num_heads: number of unique token labels
    :param is_training: if set to True, the current phase is a training one (rather than testing)
    :param dropout: the keep_probs value for the dropout
    :param sentence_lengths: the true sentence lengths, used for masking
    :param use_residual_connection: if set to True, a residual connection is added to the inputs
    :param token_scoring_method: can be either max, sum or avg
    :return sentence_scores: 2D floats of shape [B x num_sentence_labels]
    :return sentence_predictions: predicted labels for each sentence in the batch; ints of shape [B]
    :return token_scores: 3D floats of shape [B x M x num_heads]
    :return token_predictions: predicted labels for each token in each sentence; ints of shape [B x M]
    :return token_probabilities: the token scores normalized across the axis
    """
    with tf.variable_scope("multi_head_attention_original_variant"):
        num_units = inputs.get_shape().as_list()[-1]
        if num_units % num_heads != 0:
            num_units = ceil(num_units / num_heads) * num_heads
            inputs = tf.layers.dense(inputs, num_units)  # [B x M x num_units]

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

        # Mask out the keys, queries and values: replace with 0 all the token
        # positions between the true sentence length and the maximum sentence length.
        multiplication_mask = tf.tile(
            input=tf.expand_dims(tf.sequence_mask(sentence_lengths), axis=-1),
            multiples=[1, 1, num_units])  # [B, M, num_units]
        queries = tf.where(multiplication_mask, queries, tf.zeros_like(queries))
        keys = tf.where(multiplication_mask, keys, tf.zeros_like(keys))
        values = tf.where(multiplication_mask, values, tf.zeros_like(values))

        # Split and concat
        queries = tf.concat(tf.split(queries, num_heads, axis=2), axis=0)  # [(heads * B) x M x (num_units / heads)]
        keys = tf.concat(tf.split(keys, num_heads, axis=2), axis=0)  # [(heads * B) x M x (num_units / heads)]
        values = tf.concat(tf.split(values, num_heads, axis=2), axis=0)  # [(heads * B) x M x (num_units / heads)]

        # Transpose multiplication and scale
        attention_evidence = tf.matmul(queries, tf.transpose(keys, [0, 2, 1]))  # [(heads * B) x M x M]
        attention_evidence = tf.math.divide(attention_evidence, tf.constant(num_units ** 0.5))

        # Mask invalid columns (with values of -infinity), based on rows that have 0 sum.
        attention_evidence_masked = mask(attention_evidence, queries, keys, type="key")

        # Apply a non-linear layer to obtain (un-normalized) attention weights.
        if attention_activation == "soft":
            attention_weights = tf.nn.sigmoid(attention_evidence_masked)
        elif attention_activation == "sharp":
            attention_weights = tf.math.exp(attention_evidence_masked)
        elif attention_activation == "linear":
            attention_weights = attention_evidence_masked
        elif attention_activation == "softmax":
            attention_weights = tf.nn.softmax(attention_evidence_masked)
        else:
            raise ValueError("Unknown/unsupported activation for attention activation: %s."
                             % attention_activation)

        # Normalize attention weights. Will still be of shape [(heads * B) x M x M].
        if attention_activation != "softmax":
            attention_weights /= tf.reduce_sum(attention_weights, axis=-1, keep_dims=True)

        # Mask invalid rows (with values of 0), based on columns that have 0 sum.
        attention_weights = mask(attention_weights, queries, keys, type="query")

        # Apply a dropout layer.
        dropout_attention = (dropout * tf.cast(is_training, tf.float32)
                             + (1.0 - tf.cast(is_training, tf.float32)))

        attention_weights = tf.nn.dropout(
            attention_weights, dropout_attention,
            name="dropout_transducer_attention_weights")    # [(heads * B) x M x M]

        product = tf.matmul(attention_weights, values)  # [(heads * B) x M x (num_units / heads)]
        product = tf.concat(tf.split(product, num_heads), axis=2)  # [B x M x num_units]

        # Add a residual connection, followed by layer normalization.
        if use_residual_connection:
            product += inputs
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

        # Obtain token scores from the attention weights.
        # The token scores will have shape [(num_heads * B) x M x 1].
        if token_scoring_method == "sum":
            token_scores = tf.expand_dims(tf.reduce_sum(attention_evidence, axis=1), axis=2)
        elif token_scoring_method == "max":
            token_scores = tf.expand_dims(tf.reduce_max(attention_evidence, axis=1), axis=2)
        elif token_scoring_method == "avg":
            token_scores = tf.expand_dims(tf.reduce_mean(attention_evidence, axis=1), axis=2)
        else:
            raise ValueError("Unknown/unsupported token scoring method: %s" % token_scoring_method)
        token_scores = tf.concat(tf.split(token_scores, num_heads), axis=2)  # [B x M x heads]
        token_probabilities = tf.nn.softmax(token_scores)
        token_predictions = tf.argmax(token_probabilities, axis=2, output_type=tf.int32)  # [B x M]

        return sentence_scores, sentence_predictions, token_scores, token_predictions, \
               token_probabilities, attention_weights, product, token_probabilities


def multi_head_attention_with_scores_from_separate_heads(
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
        scoring_activation=None,
        separate_heads_for_sentence_scores=True):
    """
    Compute multi-head attention (mainly inspired by the transformer architecture).
    This version of the implementation applies masking at several levels:
        * first, the keys, queries and values so that the matrix multiplications
          are performed only between meaningful positions
        * second, the attention evidence values of 0 should be replaced with -infinity
          so that when applying a non-linear layer, the resulted value is very close to 0.
        * third, when obtaining the token probabilities (by normalizing across the scores),
          division masking is performed (a value of 0 should be attributed to all 0 sums).
    The masking performed before computing a primary/secondary loss is preserved.
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
    :param scoring_activation: used in computing the sentence scores from the token scores (per-head)
    :param separate_heads_for_sentence_scores: boolean value; when set to False, all heads
    are used to obtain the sentence scores; when set to True, the default and non-default heads
    from the token scores are used to obtain the sentence scores.
    :return sentence_scores: 2D floats of shape [B x num_sentence_labels]
    :return sentence_predictions: predicted labels for each sentence in the batch; ints of shape [B]
    :return token_scores: 3D floats of shape [B x M x num_heads]
    :return token_predictions: predicted labels for each token in each sentence; ints of shape [B x M]
    """
    with tf.variable_scope("multi_head_attention_variant_with_separate_scores"):
        num_units = inputs.get_shape().as_list()[-1]
        if num_units % num_heads != 0:
            num_units = ceil(num_units / num_heads) * num_heads
            inputs = tf.layers.dense(inputs, num_units)  # [B x M x num_units]

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

        # Mask out the keys, queries and values: replace with 0 all the token
        # positions between the true sentence length and the maximum sentence length.
        multiplication_mask = tf.tile(
            input=tf.expand_dims(tf.sequence_mask(sentence_lengths), axis=-1),
            multiples=[1, 1, num_units])  # [B, M, num_units]
        queries = tf.where(multiplication_mask, queries, tf.zeros_like(queries))
        keys = tf.where(multiplication_mask, keys, tf.zeros_like(keys))
        values = tf.where(multiplication_mask, values, tf.zeros_like(values))

        # Split and concat
        queries = tf.concat(tf.split(queries, num_heads, axis=2), axis=0)  # [(heads * B) x M x (num_units / heads)]
        keys = tf.concat(tf.split(keys, num_heads, axis=2), axis=0)  # [(heads * B) x M x (num_units / heads)]

        # Transpose multiplication and scale
        attention_evidence = tf.matmul(queries, tf.transpose(keys, [0, 2, 1]))  # [(heads * B) x M x M]
        attention_evidence = tf.math.divide(attention_evidence, tf.constant(num_units ** 0.5))

        # Mask invalid columns (with values of -infinity), based on rows that have 0 sum.
        attention_evidence_masked = mask(attention_evidence, queries, keys, type="key")

        # Apply a non-linear layer to obtain (un-normalized) attention weights.
        if attention_activation == "soft":
            attention_weights = tf.nn.sigmoid(attention_evidence_masked)
        elif attention_activation == "sharp":
            attention_weights = tf.math.exp(attention_evidence_masked)
        elif attention_activation == "linear":
            attention_weights = attention_evidence_masked
        elif attention_activation == "softmax":
            attention_weights = tf.nn.softmax(attention_evidence_masked)
        else:
            raise ValueError("Unknown/unsupported activation for attention activation: %s."
                             % attention_activation)

        # Normalize attention weights. Will still be of shape [(heads * B) x M x M].
        if attention_activation != "softmax":
            attention_weights /= tf.reduce_sum(attention_weights, axis=-1, keep_dims=True)

        # Mask invalid rows (with values of 0), based on columns that have 0 sum.
        attention_weights = mask(attention_weights, queries, keys, type="query")

        # Apply a dropout layer.
        dropout_attention = (dropout * tf.cast(is_training, tf.float32)
                             + (1.0 - tf.cast(is_training, tf.float32)))

        attention_weights = tf.nn.dropout(
            attention_weights, dropout_attention,
            name="dropout_transducer_attention_weights")  # [(heads * B) x M x M]

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
        token_probabilities = division_masking(
            inputs=token_scores, axis=1, multiplies=[1, num_heads, 1])  # [B, num_heads, M]
        token_predictions = tf.argmax(token_probabilities, axis=1, output_type=tf.int32)  # [B, M]

        # token_weights = token_scores / tf.reduce_sum(token_scores, axis=-1, keep_dims=True)  # [B, num_heads, M]

        # Obtain the sentence scores as a weighted sum between the inputs and the attention weights.
        weighted_sum_representation = tf.matmul(token_scores, values)  # [B, num_heads, num_units]
        if normalize_sentence:
            weighted_sum_representation = layer_normalization(weighted_sum_representation)

        if separate_heads_for_sentence_scores:
            # Get the sentence representations corresponding to the default head.
            default_head = tf.gather(
                weighted_sum_representation,
                indices=[0], axis=1)  # [B, 1, num_units]

            # Get the sentence representations corresponding to the default head.
            non_default_heads = tf.gather(
                weighted_sum_representation,
                indices=[i for i in range(1, num_heads)], axis=1)  # [B, num_heads - 1, num_units]

            # Project onto one unit (corresponding to the default sentence label score).
            sentence_default_scores = tf.layers.dense(
                default_head, units=1,
                activation=scoring_activation, kernel_initializer=initializer,
                name="ff_sentence_default_scores")  # [B, 1, 1]
            sentence_default_scores = tf.squeeze(sentence_default_scores, axis=1)  # [B, 1]

            # Project onto num_sentence_labels - 1 units, corresponding to
            # the non-default sentence label scores.
            sentence_non_default_scores = tf.layers.dense(
                non_default_heads, units=num_sentence_labels-1,
                activation=scoring_activation, kernel_initializer=initializer,
                name="ff_sentence_nondefault_scores")  # [B, num_heads - 1, num_sentence_labels-1]
            sentence_non_default_scores = tf.reduce_mean(sentence_non_default_scores, axis=1)  # [B, num_sent_labels-1]

            """
            sentence_default_score = tf.expand_dims(processed_tensor[:, 0], axis=-1)
            sentence_non_default_scores = tf.layers.dense(
                inputs=processed_tensor[:, 1:], units=(num_sentence_labels-1),
                activation=scoring_activation, kernel_initializer=initializer,
                name="ff_sentence_nondefault_scores")
            """
            sentence_scores = tf.concat(
                [sentence_default_scores, sentence_non_default_scores],
                axis=-1, name="sentence_scores_concatenation")  # [B, num_sent_labels]
        else:
            processed_tensor = tf.layers.dense(
                inputs=weighted_sum_representation, units=num_sentence_labels,
                activation=scoring_activation, kernel_initializer=initializer,
                name="ff_sentence_scores")  # [B, num_heads, num_unique_sent_labels]

            sentence_scores = tf.reduce_sum(processed_tensor, axis=1)  # [B, num_sent_labels]

        sentence_probabilities = tf.nn.softmax(sentence_scores)
        sentence_predictions = tf.argmax(sentence_probabilities, axis=1)  # [B]

        token_scores = tf.transpose(token_scores, [0, 2, 1])  # [B, M, num_heads]
        token_probabilities = tf.transpose(token_probabilities, [0, 2, 1])  # [B, M, num_heads]

        return sentence_scores, sentence_predictions, token_scores, token_predictions, token_probabilities, \
               token_probabilities, weighted_sum_representation, sentence_scores


def compute_scores_from_additive_attention(
        inputs,
        initializer,
        attention_activation,
        sentence_lengths,
        attention_size=50,
        hidden_units=200):
    """
    Compute token and sentence scores from a single-head additive attention mechanism.
    :param inputs: 3D floats of shape [B x M x E]
    :param initializer: type of initializer (best if glorot or xavier)
    :param attention_activation: type of attention activation (linear, softmax or sigmoid)
    :param sentence_lengths: 2D ints of shape [B x M]
    :param attention_size: number of units to use for the attention evidence
    :param hidden_units: number of units to use for the processed vector
    :return sentence_scores: result of the attention * input; floats of shape [B]
    :return token_scores: result of the un-normalized attention weights; floats of shape [B x M]
    :return attention_weights: 2D floats of shape [B x M] of normalized token_scores
    """
    with tf.variable_scope("compute_classic_single_head_attention"):
        attention_evidence = tf.layers.dense(
            inputs=inputs, units=attention_size,
            activation=tf.tanh, kernel_initializer=initializer)  # [B, M, attention_size]
        attention_weights = tf.layers.dense(
            inputs=attention_evidence, units=1,
            kernel_initializer=initializer)  # [B, M, 1]
        attention_weights = tf.squeeze(attention_weights, axis=-1)  # [B, M]

        # Obtain the un-normalized attention weights.
        if attention_activation == "soft":
            attention_weights = tf.nn.sigmoid(attention_weights)
        elif attention_activation == "sharp":
            attention_weights = tf.exp(attention_weights)
        elif attention_activation == "linear":
            attention_weights = attention_weights
        else:
            raise ValueError("Unknown/unsupported activation for attention: %s"
                             % attention_activation)

        attention_weights = tf.where(
            tf.sequence_mask(sentence_lengths),
            attention_weights, tf.zeros_like(attention_weights))
        token_scores = attention_weights  # [B, M]

        # Obtain the normalized attention weights (they will also be sentence weights).
        attention_weights = attention_weights / tf.reduce_sum(
            attention_weights, axis=1, keep_dims=True)  # [B, M]
        product = inputs * tf.expand_dims(attention_weights, axis=-1)  # [B, M, E]
        processed_tensor = tf.reduce_sum(product, axis=1)  # [B, E]

        if hidden_units > 0:
            processed_tensor = tf.layers.dense(
                inputs=processed_tensor, units=hidden_units,
                activation=tf.tanh, kernel_initializer=initializer)  # [B, hidden_units]

        sentence_scores = tf.layers.dense(
            inputs=processed_tensor, units=1,
            activation=tf.sigmoid, kernel_initializer=initializer,
            name="output_sent_single_head_ff")  # [B, 1]
        sentence_scores = tf.squeeze(sentence_scores, axis=-1)
        return sentence_scores, token_scores, attention_weights


def compute_scores_from_scaled_dot_product_attention(
        inputs,
        initializer,
        attention_activation,
        sentence_lengths,
        token_scoring_method):
    """
    Compute token and sentence scores from a single-head scaled dot product attention mechanism.
    :param inputs: 3D floats of shape [B x M x E]
    :param initializer: type of initializer (best if glorot or xavier)
    :param attention_activation: type of attention activation (linear, softmax or sigmoid)
    :param sentence_lengths: 2D ints of shape [B x M]
    :param token_scoring_method: can be either max, sum or avg
    :return sentence_scores: 2D floats of shape [B x num_sentence_labels]
    :return token_scores: 2D floats of shape [B x M]
    :return token_probabilities: 2D floats of shape [B x M] of normalized token_scores
    """
    with tf.variable_scope("compute_transformer_single_head_attention"):
        num_units = inputs.get_shape().as_list()[-1]

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

        # Scaled dot-product attention.
        attention_evidence = tf.matmul(queries, tf.transpose(keys, [0, 2, 1]))  # [B, M, M]
        attention_evidence = tf.math.divide(attention_evidence, tf.constant(num_units ** 0.5))

        # Mask out the 0 values in the attention_evidence before proceeding to the non-linear activation.
        attention_evidence = tf.where(
            tf.equal(attention_evidence, 0),
            tf.ones_like(attention_evidence) * (-2 ** 32 + 1),
            attention_evidence)  # [B, M, M]

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

        # Obtain the token scores from the attention weights.
        # The token_scores below will have shape [B, M].
        if token_scoring_method == "sum":
            token_scores = tf.reduce_sum(attention_weights, axis=1)
        elif token_scoring_method == "max":
            token_scores = tf.reduce_max(attention_weights, axis=1)
        elif token_scoring_method == "avg":
            token_scores = tf.reduce_mean(attention_weights, axis=1)
        else:
            raise ValueError("Unknown/unsupported token scoring method: %s"
                             % token_scoring_method)

        # Mask the token scores
        token_scores_mask = tf.sequence_mask(sentence_lengths)  # [B, M]

        token_scores = tf.where(token_scores_mask, token_scores, tf.zeros_like(token_scores))

        token_probabilities = division_masking(
            inputs=token_scores, axis=-1, multiplies=[1, tf.shape(token_scores)[1]])  # [B, M]
        token_probabilities = tf.expand_dims(token_probabilities, axis=1)  # [B, 1, M]

        # Obtain the sentence scores as a weighted sum between the inputs and the attention weights.
        weighted_sum_representation = tf.matmul(token_probabilities, values)  # [B, 1, num_units]
        sentence_scores = tf.reduce_sum(weighted_sum_representation, axis=-1)  # [B, 1]
        sentence_scores = tf.squeeze(sentence_scores, axis=-1)
        return sentence_scores, token_scores, token_probabilities


def single_head_attention_multiple_transformations(
        inputs,
        initializer,
        attention_activation,
        num_sentence_labels,
        num_heads,
        sentence_lengths,
        token_scoring_method,
        scoring_activation=None,
        how_to_compute_attention="dot",
        separate_heads_for_sentence_scores=True):
    """
    Compute token and sentence scores using a single-head attention mechanism,
    which can either be additive (mainly inspired by the single-head binary-label
    method above, as in Rei and Søgaard's paper https://arxiv.org/pdf/1811.05949.pdf)
    or a scaled-dot product version (inspired by the transformer, but with just one head).
    Then, use these scores to obtain predictions at both granularities.
    :param inputs: 3D floats of shape [B x M x E]
    :param initializer: type of initializer (best if glorot or xavier)
    :param attention_activation
    :param num_sentence_labels: number of unique sentence labels
    :param num_heads: number of unique token labels
    :param sentence_lengths: the true sentence lengths, used for masking
    :param token_scoring_method
    :param scoring_activation: activation used for scoring, default is None.
    :param how_to_compute_attention: compute attention in the classic way (Marek) or as in transformer
    :param separate_heads_for_sentence_scores: boolean value; when set to False, all heads
    are used to obtain the sentence scores; when set to True, the default and non-default heads
    from the token scores are used to obtain the sentence scores.
    :return sentence_scores: 2D floats of shape [B x num_sentence_labels]
    :return sentence_predictions: predicted labels for each sentence in the batch; ints of shape [B]
    :return token_scores: 3D floats of shape [B x M x num_heads]
    :return token_predictions: predicted labels for each token in each sentence; ints of shape [B x M]
    """
    with tf.variable_scope("transformer_single_heads_multi_attention"):
        token_scores_per_head = []
        token_probabilities_per_head = []
        sentence_scores_per_head = []
        for i in range(num_heads):
            with tf.variable_scope("num_head_{}".format(i), reuse=tf.AUTO_REUSE):
                if how_to_compute_attention == "additive":
                    sentence_scores_head_i, token_scores_head_i, token_prob_head_i = \
                        compute_scores_from_additive_attention(
                            inputs=inputs, initializer=initializer,
                            attention_activation=attention_activation,
                            sentence_lengths=sentence_lengths)
                elif how_to_compute_attention == "dot":
                    sentence_scores_head_i, token_scores_head_i, token_prob_head_i = \
                        compute_scores_from_scaled_dot_product_attention(
                            inputs=inputs, initializer=initializer,
                            attention_activation=attention_activation,
                            sentence_lengths=sentence_lengths,
                            token_scoring_method=token_scoring_method)
                else:
                    raise ValueError("Unknown/unsupported way of computing the attention: %s"
                                     % how_to_compute_attention)
                sentence_scores_per_head.append(sentence_scores_head_i)
                token_scores_per_head.append(token_scores_head_i)
                token_probabilities_per_head.append(token_prob_head_i)

        sentence_scores = tf.stack(sentence_scores_per_head, axis=-1)  # [B, num_heads]

        if separate_heads_for_sentence_scores:
            sentence_default_score = tf.layers.dense(
                inputs=tf.expand_dims(sentence_scores[:, 0], axis=-1), units=1,
                activation=scoring_activation, kernel_initializer=initializer,
                name="ff_non_default_sentence_scores")
            sentence_non_default_scores = tf.layers.dense(
                inputs=sentence_scores[:, 1:], units=num_sentence_labels-1,
                activation=scoring_activation, kernel_initializer=initializer,
                name="ff_default_sentence_scores")
            sentence_scores = tf.concat([sentence_default_score, sentence_non_default_scores],
                                        axis=-1, name="sentence_scores_concatenation")
        else:
            sentence_scores = tf.layers.dense(
                inputs=sentence_scores, units=num_sentence_labels,
                activation=scoring_activation, kernel_initializer=initializer,
                name="ff_sentence_scores")  # [B, num_sentence_labels]

        sentence_probabilities = tf.nn.softmax(sentence_scores)
        sentence_predictions = tf.argmax(sentence_probabilities, axis=1)  # [B]

        token_scores = tf.stack(token_scores_per_head, axis=-1)  # [B, M, num_heads]
        token_probabilities = tf.stack(token_scores_per_head, axis=-1)  # [B, M, num_heads]
        token_predictions = tf.argmax(token_probabilities, axis=-1)  # [B, M]

        return sentence_scores, sentence_predictions, token_scores, token_predictions, token_probabilities

