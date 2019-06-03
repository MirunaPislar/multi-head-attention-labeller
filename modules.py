from math import ceil
import tensorflow as tf


def layer_normalization(layer, epsilon=1e-8):
    """
    Implements layer normalization.
    :param layer: has 2-dimensional, the first dimension is the batch_size
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
    Implements label smoothing. This prevents the model from becoming
    over-confident about its predictions and thus, less prone to overfitting.
    Label smoothing regularizes the model and makes it more adaptable.
    :param labels: 3D tensor with the last dimension as the number of labels
    :param epsilon: smoothing rate
    :return: smoothed labels
    """
    num_labels = labels.get_shape().as_list()[-1]
    return ((1 - epsilon) * labels) + (epsilon / num_labels)


def mask(inputs, queries=None, keys=None, mask_type=None):
    """
    Generates masks and apply them to 3D inputs.
    inputs: 3D tensor. [B, M, M]
    queries: 3D tensor. [B, M, E]
    keys: 3D tensor. [B, M, E]
    """
    padding_num = -2 ** 32 + 1
    if "key" in mask_type:
        masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # [B, M]
        masks = tf.expand_dims(masks, axis=1)  # [B, 1, M]
        masks = tf.tile(masks, [1, tf.shape(queries)[1], 1])  # [B, M, M]
        paddings = tf.ones_like(inputs) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)  # [B, M, M]
    elif "query" in mask_type:
        masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # [B, M]
        masks = tf.expand_dims(masks, axis=-1)  # [B, M, 1]
        masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # [B, M, M]
        outputs = inputs * masks
    else:
        raise ValueError("Unknown mask type: %s. You need to choose "
                         "between \"keys\" and \"query\"." % mask_type)
    return outputs


def mask_2(inputs, queries=None, keys=None, mask_type=None):
    """
    Generates masks and apply them to 4D inputs.
    inputs: 3D tensor. [H, B, M, M]
    queries: 3D tensor. [H, B, M, E]
    keys: 3D tensor. [H, B, M, E]
    """
    padding_num = -2 ** 32 + 1
    if "key" in mask_type:
        masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # [H, B, M]
        masks = tf.expand_dims(masks, axis=2)  # [H, B, 1, M]
        masks = tf.tile(masks, [1, 1, tf.shape(queries)[2], 1])  # [H, B, M, M]
        paddings = tf.ones_like(inputs) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)  # [H, B, M, M]
    elif "query" in mask_type:
        masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # [H, B, M]
        masks = tf.expand_dims(masks, axis=-1)  # [H, B, M, 1]
        masks = tf.tile(masks, [1, 1, 1, tf.shape(keys)[2]])  # [H, B, M, M]
        outputs = inputs * masks
    else:
        raise ValueError("Unknown mask type: %s. You need to choose "
                         "between \"keys\" and \"query\"." % mask_type)
    return outputs


def cosine_distance_loss(inputs, take_abs=False):
    """
    Computes the cosine pairwise distance loss between the input heads.
    :param inputs: expects tensor with its last two dimensions [*, H, E],
    where H = num heads and E = arbitrary vector dimension.
    :param take_abs: take the absolute value of the cosine similarity; this
    has the effect of switching from [-1, 1] to [0, 1], with the minimum at 0,
    i.e. when the vectors are orthogonal, which is what we want.
    However, this might not be differentiable at 0.
    :return: loss of the cosine distance between any 2 pairs of head vectors.
    """
    with tf.variable_scope("cosine_distance_loss"):
        # Calculate the cosine similarity and cosine distance.
        # The goal is to maximize the cosine distance.
        normalized_inputs = tf.nn.l2_normalize(inputs, axis=-1)
        permutation = list(range(len(inputs.get_shape().as_list())))
        permutation[-1], permutation[-2] = permutation[-2], permutation[-1]
        cos_similarity = tf.matmul(
            normalized_inputs, tf.transpose(normalized_inputs, permutation))

        # Mask the lower diagonal matrix.
        ones = tf.ones_like(cos_similarity)
        mask_upper = tf.matrix_band_part(ones, 0, -1)  # upper triangular part
        mask_diagonal = tf.matrix_band_part(ones, 0, 0)  # diagonal
        mask_matrix = tf.cast(mask_upper - mask_diagonal, dtype=tf.bool)

        upper_triangular_flat = tf.boolean_mask(cos_similarity, mask_matrix)

        if take_abs:
            return tf.reduce_mean(tf.math.abs(upper_triangular_flat))
        else:
            return tf.reduce_mean(upper_triangular_flat)


def single_head_attention_binary_labels(
        inputs,
        initializer,
        attention_size,
        sentence_lengths,
        hidden_units):
    """
    Computes single-head attention (just normal, vanilla, soft attention).
    :param inputs: 3D floats of shape [B, M, E]
    :param initializer: type of initializer (best if Glorot or Xavier)
    :param attention_size: number of units to use for the attention evidence
    :param sentence_lengths: 2D ints of shape [B, M]
    :param hidden_units: number of units to use for the processed sent tensor
    :return sentence_scores: result of the attention * input; floats of shape [B]
    :return sentence_predictions: predicted labels for each sentence in the batch; ints of shape [B]
    :return token_scores: result of the un-normalized attention weights; floats of shape [B, M]
    :return token_predictions: predicted labels for each token in each sentence; ints of shape [B, M]
    """
    with tf.variable_scope("single_head_attention_binary_labels"):
        attention_evidence = tf.layers.dense(
            inputs=inputs, units=attention_size,
            activation=tf.tanh, kernel_initializer=initializer)  # [B, M, attention_size]
        attention_weights = tf.layers.dense(
            inputs=attention_evidence, units=1,
            kernel_initializer=initializer)  # [B, M, 1]
        attention_weights = tf.squeeze(attention_weights, axis=-1)  # [B, M]
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
        sentence_scores = tf.reshape(
            sentence_scores, shape=[tf.shape(processed_tensor)[0]])  # [B]
        sentence_predictions = tf.where(
            tf.greater_equal(sentence_scores, 0.5),
            tf.ones_like(sentence_scores, dtype=tf.int32),
            tf.zeros_like(sentence_scores, dtype=tf.int32))     # [B]
        return sentence_scores, sentence_predictions, token_scores, token_predictions


def baseline_lstm_last_contexts(
        last_token_contexts,
        last_context,
        initializer,
        scoring_activation,
        sentence_lengths,
        hidden_units,
        num_sentence_labels,
        num_token_labels):
    """
    Computes token and sentence scores/predictions solely from the last LSTM contexts.
    vectors that the Bi-LSTM has produced. Works for flexible no. of labels.
    :param last_token_contexts: the (concatenated) Bi-LSTM outputs per-token.
    :param last_context: the (concatenated) Bi-LSTM final state.
    :param initializer: type of initializer (best if Glorot or Xavier)
    :param scoring_activation: used in computing the sentence scores from the token scores (per-head)
    :param sentence_lengths: 2D ints of shape [B, M]
    :param hidden_units: number of units to use for the processed sentence tensor
    :param num_sentence_labels: number of unique sentence labels
    :param num_token_labels: number of unique token labels
    :return sentence_scores: 2D floats of shape [B, num_sentence_labels]
    :return sentence_predictions: predicted labels for each sentence in the batch; ints of shape [B]
    :return token_scores: 3D floats of shape [B, M, num_token_labels]
    :return token_predictions: predicted labels for each token in each sentence; ints of shape [B, M]
    :return: attention weights will be a tensor of zeros of shape [B, M, num_token_labels].
    """
    with tf.variable_scope("baseline_lstm_last_contexts"):
        if hidden_units > 0:
            processed_tensor = tf.layers.dense(
                last_context, units=hidden_units,
                activation=tf.tanh, kernel_initializer=initializer)
            token_scores = tf.layers.dense(
                last_token_contexts, units=hidden_units,
                activation=tf.tanh, kernel_initializer=initializer)
        else:
            processed_tensor = last_context
            token_scores = last_token_contexts

        sentence_scores = tf.layers.dense(
            processed_tensor, units=num_sentence_labels,
            activation=scoring_activation, kernel_initializer=initializer,
            name="sentence_scores_lstm_ff")  # [B, num_sentence_labels]
        sentence_probabilities = tf.nn.softmax(sentence_scores, axis=-1)
        sentence_predictions = tf.argmax(sentence_probabilities, axis=-1)  # [B]

        token_scores = tf.layers.dense(
            token_scores, units=num_token_labels,
            activation=scoring_activation, kernel_initializer=initializer,
            name="token_scores_lstm_ff")  # [B, M, num_token_labels]

        masked_sentence_lengths = tf.tile(
            input=tf.expand_dims(
                tf.sequence_mask(sentence_lengths), axis=-1),
            multiples=[1, 1, num_token_labels])
        token_scores = tf.where(
            masked_sentence_lengths,
            token_scores,
            tf.zeros_like(token_scores))  # [B, M, num_token_labels]
        token_probabilities = tf.nn.softmax(token_scores, axis=-1)
        token_predictions = tf.argmax(token_probabilities, axis=-1)
        attention_weights = tf.zeros_like(token_scores)

        return sentence_scores, sentence_predictions, token_scores, token_predictions, \
            token_probabilities, sentence_probabilities, attention_weights


def single_head_attention_multiple_labels(
        inputs,
        initializer,
        attention_activation,
        attention_size,
        sentence_lengths,
        hidden_units,
        num_sentence_labels,
        num_token_labels):
    """
    Computes single-head attention, but adapt it (naively) to make it work for multiple labels.
    :param inputs: 3D floats of shape [B, M, E]
    :param initializer: type of initializer (best if Glorot or Xavier)
    :param attention_activation: type of attention activation (soft, sharp, linear, etc)
    :param attention_size: number of units to use for the attention evidence
    :param sentence_lengths: 2D ints of shape [B, M]
    :param hidden_units: number of units to use for the processed sent tensor
    :param num_sentence_labels: number of unique sentence labels
    :param num_token_labels: number of unique token labels
    :return sentence_scores: 2D floats of shape [B, num_sentence_labels]
    :return sentence_predictions: predicted labels for each sentence in the batch; ints of shape [B]
    :return token_scores: 3D floats of shape [B, M, num_token_labels]
    :return token_predictions: predicted labels for each token in each sentence; ints of shape [B, M]
    """
    with tf.variable_scope("SHA_multiple_labels"):
        attention_evidence = tf.layers.dense(
            inputs=inputs, units=attention_size,
            activation=tf.tanh, kernel_initializer=initializer)  # [B, M, attention_size]

        attention_evidence = tf.layers.dense(
            inputs=attention_evidence, units=1,
            kernel_initializer=initializer)  # [B, M, 1]
        attention_evidence = tf.squeeze(attention_evidence, axis=-1)  # [B, M]

        # Apply a non-linear layer to obtain (un-normalized) attention weights.
        if attention_activation == "soft":
            attention_weights = tf.nn.sigmoid(attention_evidence)
        elif attention_activation == "sharp":
            attention_weights = tf.math.exp(attention_evidence)
        elif attention_activation == "linear":
            attention_weights = attention_evidence
        elif attention_activation == "softmax":
            attention_weights = tf.nn.softmax(attention_evidence)
        else:
            raise ValueError("Unknown/unsupported activation for attention activation: %s."
                             % attention_activation)

        # Mask attention weights.
        attention_weights = tf.where(
            tf.sequence_mask(sentence_lengths),
            attention_weights, tf.zeros_like(attention_weights))
        attention_weights_unnormalized = attention_weights

        # Normalize attention weights.
        if attention_activation != "softmax":
            attention_weights = attention_weights / tf.reduce_sum(
                attention_weights, axis=-1, keep_dims=True)  # [B, M]

        token_scores = tf.layers.dense(
            inputs=tf.expand_dims(attention_weights_unnormalized, -1),
            units=num_token_labels,
            kernel_initializer=initializer,
            name="output_single_head_token_scores_ff")  # [B, M, num_token_labels]
        token_probabilities = tf.nn.softmax(token_scores)
        token_predictions = tf.argmax(token_probabilities,
                                      axis=2, output_type=tf.int32)  # [B, M]

        product = inputs * tf.expand_dims(attention_weights, axis=-1)  # [B, M, E]
        processed_tensor = tf.reduce_sum(product, axis=1)  # [B, E]

        if hidden_units > 0:
            processed_tensor = tf.layers.dense(
                inputs=processed_tensor, units=hidden_units,
                activation=tf.tanh, kernel_initializer=initializer)  # [B, hidden_units]

        sentence_scores = tf.layers.dense(
            inputs=processed_tensor, units=num_sentence_labels,
            kernel_initializer=initializer,
            name="output_multi_sent_specified_scores_ff")  # [B, num_unique_sent_labels]
        sentence_probabilities = tf.nn.softmax(sentence_scores, axis=-1)
        sentence_predictions = tf.argmax(sentence_probabilities, axis=-1)  # [B]
        return sentence_scores, sentence_predictions, token_scores, token_predictions, \
            token_probabilities, sentence_probabilities, attention_weights


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
    Computes multi-head attention (mainly inspired by the transformer architecture).
    This method does not take into account any masking at any level.
    All the masking will be performed before computing a primary/secondary loss.
    :param inputs: 3D floats of shape [B, M, E]
    :param initializer: type of initializer (best if Glorot or Xavier)
    :param attention_activation: type of attention activation (linear, softmax or sigmoid)
    :param hidden_units: number of units to use for the processed sent tensor
    :param num_sentence_labels: number of unique sentence labels
    :param num_heads: number of unique token labels
    :param is_training: if set to True, the current phase is a training one (rather than testing)
    :param dropout: the keep_probs value for the dropout
    :param sentence_lengths: the true sentence lengths, used for masking
    :param use_residual_connection: if set to True, a residual connection is added to the inputs
    :param token_scoring_method: can be either max, sum or avg
    :return sentence_scores: 2D floats of shape [B, num_sentence_labels]
    :return sentence_predictions: predicted labels for each sentence in the batch; ints of shape [B]
    :return token_scores: 3D floats of shape [B, M, num_heads]
    :return token_predictions: predicted labels for each token in each sentence; ints of shape [B, M]
    :return token_probabilities: the token scores normalized across the axis
    """
    with tf.variable_scope("MHA_sentence_scores_from_shared_heads"):
        num_units = inputs.get_shape().as_list()[-1]
        if num_units % num_heads != 0:
            num_units = ceil(num_units / num_heads) * num_heads
            inputs = tf.layers.dense(inputs, num_units)  # [B, M, num_units]

        # Project to get the queries, keys, and values.
        queries = tf.layers.dense(
            inputs, num_units, activation=tf.tanh,
            kernel_initializer=initializer)  # [B, M, num_units]
        keys = tf.layers.dense(
            inputs, num_units, activation=tf.tanh,
            kernel_initializer=initializer)  # [B, M, num_units]
        values = tf.layers.dense(
            inputs, num_units, activation=tf.tanh,
            kernel_initializer=initializer)  # [B, M, num_units]

        # Mask out the keys, queries and values: replace with 0 all the token
        # positions between the true and the maximum sentence length.
        multiplication_mask = tf.tile(
            input=tf.expand_dims(tf.sequence_mask(sentence_lengths), axis=-1),
            multiples=[1, 1, num_units])  # [B, M, num_units]
        queries = tf.where(multiplication_mask, queries, tf.zeros_like(queries))
        keys = tf.where(multiplication_mask, keys, tf.zeros_like(keys))
        values = tf.where(multiplication_mask, values, tf.zeros_like(values))

        # Split and concat as many projections as the number of heads.
        queries = tf.concat(
            tf.split(queries, num_heads, axis=2),
            axis=0)  # [B*num_heads, M, num_units/num_heads]
        keys = tf.concat(
            tf.split(keys, num_heads, axis=2),
            axis=0)  # [B*num_heads, M, num_units/num_heads]
        values = tf.concat(
            tf.split(values, num_heads, axis=2),
            axis=0)  # [B*num_heads, M, num_units/num_heads]

        # Transpose multiplication and scale
        attention_evidence = tf.matmul(
            queries, tf.transpose(keys, [0, 2, 1]))  # [B*num_heads, M, M]
        attention_evidence = tf.math.divide(
            attention_evidence, tf.constant(num_units ** 0.5))

        # Mask columns (with values of -infinity), based on rows that have 0 sum.
        attention_evidence_masked = mask(
            attention_evidence, queries, keys, mask_type="key")

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
            raise ValueError("Unknown/unsupported attention activation: %s."
                             % attention_activation)

        attention_weights_unnormalized = attention_weights

        # Normalize attention weights.
        if attention_activation != "softmax":
            attention_weights /= tf.reduce_sum(
                attention_weights, axis=-1, keep_dims=True)

        # Mask rows (with values of 0), based on columns that have 0 sum.
        attention_weights = mask(
            attention_weights, queries, keys, mask_type="query")
        attention_weights_unnormalized = mask(
            attention_weights_unnormalized, queries, keys, mask_type="query")

        # Apply a dropout layer on the attention weights.
        if dropout > 0.0:
            dropout_attention = (dropout * tf.cast(is_training, tf.float32)
                                 + (1.0 - tf.cast(is_training, tf.float32)))
            attention_weights = tf.nn.dropout(
                attention_weights, dropout_attention,
                name="dropout_attention_weights")    # [B*num_heads, M, M]

        # [B*num_heads, M, num_units/num_heads]
        product = tf.matmul(attention_weights, values)
        product = tf.concat(
            tf.split(product, num_heads), axis=2)  # [B, M, num_units]

        # Add a residual connection, followed by layer normalization.
        if use_residual_connection:
            product += inputs
            product = layer_normalization(product)  # [B, M, num_units]

        processed_tensor = tf.reduce_sum(product, axis=1)  # [B, num_units]

        if hidden_units > 0:
            processed_tensor = tf.layers.dense(
                inputs=processed_tensor, units=hidden_units,
                activation=tf.tanh, kernel_initializer=initializer)  # [B, hidden_units]

        sentence_scores = tf.layers.dense(
            inputs=processed_tensor, units=num_sentence_labels,
            kernel_initializer=initializer,
            name="output_sent_specified_scores_ff")  # [B, num_unique_sent_labels]
        sentence_probabilities = tf.nn.softmax(sentence_scores)
        sentence_predictions = tf.argmax(sentence_probabilities, axis=1)  # [B]

        # Obtain token scores from the attention weights.
        # The token scores will have shape [B*num_heads, M, 1].
        if token_scoring_method == "sum":
            token_scores = tf.expand_dims(tf.reduce_sum(
                attention_weights_unnormalized, axis=1), axis=2)
        elif token_scoring_method == "max":
            token_scores = tf.expand_dims(tf.reduce_max(
                attention_weights_unnormalized, axis=1), axis=2)
        elif token_scoring_method == "avg":
            token_scores = tf.expand_dims(tf.reduce_mean(
                attention_weights_unnormalized, axis=1), axis=2)
        elif token_scoring_method == "logsumexp":
            token_scores = tf.expand_dims(tf.reduce_logsumexp(
                attention_weights_unnormalized, axis=1), axis=2)
        else:
            raise ValueError("Unknown/unsupported token scoring method: %s"
                             % token_scoring_method)

        token_scores = tf.concat(
            tf.split(token_scores, num_heads), axis=2)  # [B, M, num_heads]
        token_probabilities = tf.nn.softmax(token_scores)
        token_predictions = tf.argmax(
            token_probabilities, axis=2, output_type=tf.int32)  # [B, M]

        attention_weights = tf.concat(
            tf.split(tf.expand_dims(attention_weights, axis=-1), num_heads),
            axis=-1)  # [B, M, M, num_heads]

        return sentence_scores, sentence_predictions, \
            token_scores, token_predictions, \
            token_probabilities, sentence_probabilities, attention_weights


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
        separate_heads=True):
    """
    Computes multi-head attention (mainly inspired by the transformer architecture).
    This version of the implementation applies masking at several levels:
        * first, the keys, queries and values so that the matrix multiplications
          are performed only between meaningful positions
        * second, the attention evidence values of 0 should be replaced with -infinity
          so that when applying a non-linear layer, the resulted value is very close to 0.
        * third, when obtaining the token probabilities (by normalizing across the scores),
          division masking is performed (a value of 0 should be attributed to all 0 sums).
    The masking performed before computing a primary/secondary loss is preserved.
    :param inputs: 3D floats of shape [B, M, E]
    :param initializer: type of initializer (best if Glorot or Xavier)
    :param attention_activation: type of attention activation (linear, softmax or sigmoid)
    :param num_sentence_labels: number of unique sentence labels
    :param num_heads: number of unique token labels
    :param is_training: if set to True, the current phase is a training one (rather than testing)
    :param dropout: the keep_probs value for the dropout
    :param sentence_lengths: the true sentence lengths, used for masking
    :param normalize_sentence: if set to True, the last weighted sentence layer is normalized
    :param token_scoring_method: can be either max, sum or avg
    :param scoring_activation: used in computing the sentence scores from the token scores (per-head)
    :param separate_heads: boolean value; when set to False, all heads
    are used to obtain the sentence scores; when set to True, the default and non-default heads
    from the token scores are used to obtain the sentence scores.
    :return sentence_scores: 2D floats of shape [B, num_sentence_labels]
    :return sentence_predictions: predicted labels for each sentence in the batch; ints of shape [B]
    :return token_scores: 3D floats of shape [B, M, num_heads]
    :return token_predictions: predicted labels for each token in each sentence; ints of shape [B, M]
    """
    with tf.variable_scope("MHA_sentence_scores_from_separate_heads"):
        num_units = inputs.get_shape().as_list()[-1]
        if num_units % num_heads != 0:
            num_units = ceil(num_units / num_heads) * num_heads
            inputs = tf.layers.dense(inputs, num_units)  # [B, M, num_units]

        # Project to get the queries, keys, and values.
        queries = tf.layers.dense(
            inputs, num_units, activation=tf.tanh,
            kernel_initializer=initializer)  # [B, M, num_units]
        keys = tf.layers.dense(
            inputs, num_units, activation=tf.tanh,
            kernel_initializer=initializer)  # [B, M, num_units]
        values = tf.layers.dense(
            inputs, num_units, activation=tf.tanh,
            kernel_initializer=initializer)  # [B, M, num_units]

        # Mask out the keys, queries and values: replace with 0 all the token
        # positions between the true and the maximum sentence length.
        multiplication_mask = tf.tile(
            input=tf.expand_dims(tf.sequence_mask(sentence_lengths), axis=-1),
            multiples=[1, 1, num_units])  # [B, M, num_units]
        queries = tf.where(multiplication_mask, queries, tf.zeros_like(queries))
        keys = tf.where(multiplication_mask, keys, tf.zeros_like(keys))
        values = tf.where(multiplication_mask, values, tf.zeros_like(values))

        # Split and concat as many projections as the number of heads.
        queries = tf.concat(
            tf.split(queries, num_heads, axis=2),
            axis=0)  # [B*num_heads, M, num_units/num_heads]
        keys = tf.concat(
            tf.split(keys, num_heads, axis=2),
            axis=0)  # [B*num_heads, M, num_units/num_heads]

        # Transpose multiplication and scale
        attention_evidence = tf.matmul(
            queries, tf.transpose(keys, [0, 2, 1]))  # [B*num_heads, M, M]
        attention_evidence = tf.math.divide(
            attention_evidence, tf.constant(num_units ** 0.5))

        # Mask columns (with values of -infinity), based on rows that have 0 sum.
        attention_evidence_masked = mask(
            attention_evidence, queries, keys, mask_type="key")

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
            raise ValueError("Unknown/unsupported attention activation: %s."
                             % attention_activation)

        # Normalize attention weights.
        if attention_activation != "softmax":
            attention_weights /= tf.reduce_sum(
                attention_weights, axis=-1, keep_dims=True)

        # Mask rows (with values of 0), based on columns that have 0 sum.
        attention_weights = mask(
            attention_weights, queries, keys, mask_type="query")

        # Apply a dropout layer on the attention weights.
        if dropout > 0.0:
            dropout_attention = (dropout * tf.cast(is_training, tf.float32)
                                 + (1.0 - tf.cast(is_training, tf.float32)))
            attention_weights = tf.nn.dropout(
                attention_weights, dropout_attention,
                name="dropout_attention_weights")  # [B*num_heads, M, M]

        # Obtain the token scores from the attention weights.
        # The token_scores below will have shape [B*num_heads, 1, M].
        if token_scoring_method == "sum":
            token_scores = tf.reduce_sum(
                attention_weights, axis=1, keep_dims=True)
        elif token_scoring_method == "max":
            token_scores = tf.reduce_max(
                attention_weights, axis=1, keep_dims=True)
        elif token_scoring_method == "avg":
            token_scores = tf.reduce_mean(
                attention_weights, axis=1, keep_dims=True)
        elif token_scoring_method == "logsumexp":
            token_scores = tf.reduce_logsumexp(
                attention_weights, axis=1, keep_dims=True)
        else:
            raise ValueError("Unknown/unsupported token scoring method: %s"
                             % token_scoring_method)

        token_scores = tf.concat(
            tf.split(token_scores, num_heads),
            axis=1)  # [B, num_heads, M]
        token_scores_normalized = division_masking(
            inputs=token_scores, axis=-1,
            multiplies=[1, 1, tf.shape(token_scores)[-1]])  # [B, num_heads, M]
        token_probabilities = tf.nn.softmax(token_scores, axis=1)
        token_predictions = tf.argmax(
            token_probabilities, axis=1, output_type=tf.int32)  # [B, M]

        # Obtain a weighted sum between the inputs and the attention weights.
        # [B, num_heads, num_units]
        weighted_sum_representation = tf.matmul(token_scores_normalized, values)

        if normalize_sentence:
            weighted_sum_representation = layer_normalization(weighted_sum_representation)

        if separate_heads:
            # Get the sentence representations corresponding to the default head.
            default_head = tf.gather(
                weighted_sum_representation,
                indices=[0], axis=1)  # [B, 1, num_units]

            # Get the sentence representations corresponding to the default head.
            non_default_heads = tf.gather(
                weighted_sum_representation,
                indices=list(range(1, num_heads)), axis=1)  # [B, num_heads-1, num_units]

            # Project onto one unit, corresponding to
            # the default sentence label score.
            sentence_default_scores = tf.layers.dense(
                default_head, units=1,
                activation=scoring_activation, kernel_initializer=initializer,
                name="sentence_default_scores_ff")  # [B, 1, 1]
            sentence_default_scores = tf.squeeze(
                sentence_default_scores, axis=-1)  # [B, 1]

            # Project onto (num_sentence_labels-1) units, corresponding to
            # the non-default sentence label scores.
            sentence_non_default_scores = tf.layers.dense(
                non_default_heads, units=num_sentence_labels-1,
                activation=scoring_activation, kernel_initializer=initializer,
                name="sentence_non_default_scores_ff")  # [B, num_heads-1, num_sentence_labels-1]

            sentence_non_default_scores = tf.reduce_mean(
                sentence_non_default_scores, axis=1)  # [B, num_sent_labels-1]

            sentence_scores = tf.concat(
                [sentence_default_scores, sentence_non_default_scores],
                axis=-1, name="sentence_scores_concatenation")  # [B, num_sent_labels]
        else:
            processed_tensor = tf.layers.dense(
                inputs=weighted_sum_representation, units=num_sentence_labels,
                activation=scoring_activation, kernel_initializer=initializer,
                name="sentence_scores_ff")  # [B, num_heads, num_unique_sent_labels]
            sentence_scores = tf.reduce_sum(
                processed_tensor, axis=1)  # [B, num_sent_labels]

        sentence_probabilities = tf.nn.softmax(sentence_scores)
        sentence_predictions = tf.argmax(sentence_probabilities, axis=1)  # [B]

        # Get token scores and probabilities of shape # [B, M, num_heads].
        token_scores = tf.transpose(token_scores, [0, 2, 1])
        token_probabilities = tf.transpose(token_probabilities, [0, 2, 1])

        attention_weights = tf.concat(
            tf.split(tf.expand_dims(attention_weights, axis=-1), num_heads),
            axis=-1)  # [B, M, M, num_heads]

        return sentence_scores, sentence_predictions, \
            token_scores, token_predictions, \
            token_probabilities, sentence_probabilities, attention_weights


def compute_scores_from_additive_attention(
        inputs,
        initializer,
        attention_activation,
        sentence_lengths,
        attention_size=50,
        hidden_units=50):
    """
    Computes token and sentence scores from a single-head additive attention mechanism.
    :param inputs: 3D floats of shape [B, M, E]
    :param initializer: type of initializer (best if Glorot or Xavier)
    :param attention_activation: type of attention activation (linear, softmax or sigmoid)
    :param sentence_lengths: 2D ints of shape [B, M]
    :param attention_size: number of units to use for the attention evidence
    :param hidden_units: number of units to use for the processed sent tensor
    :return sentence_scores: result of the attention * input; floats of shape [B]
    :return token_scores: result of the un-normalized attention weights; floats of shape [B, M]
    :return attention_weights: 2D floats of shape [B, M] of normalized token_scores
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
            raise ValueError("Unknown/unsupported attention activation: %s"
                             % attention_activation)

        attention_weights = tf.where(
            tf.sequence_mask(sentence_lengths),
            attention_weights, tf.zeros_like(attention_weights))
        token_scores = attention_weights  # [B, M]

        # Obtain the normalized attention weights (they will also be sentence weights).
        attention_weights = attention_weights / tf.reduce_sum(
            attention_weights, axis=1, keep_dims=True)  # [B, M]
        product = inputs * tf.expand_dims(attention_weights, axis=-1)  # [B, M, num_units]
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
    Computes token and sentence scores from a single-head scaled dot product attention mechanism.
    :param inputs: 3D floats of shape [B, M, E]
    :param initializer: type of initializer (best with Glorot or Xavier)
    :param attention_activation: type of attention activation: sharp (exp) or soft (sigmoid)
    :param sentence_lengths: 2D ints of shape [B, M]
    :param token_scoring_method: can be either max, sum or avg
    :return sentence_scores: 2D floats of shape [B, num_sentence_labels]
    :return token_scores: 2D floats of shape [B, M]
    :return token_probabilities: 2D floats of shape [B, M] of normalized token_scores
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

        # Mask out the keys, queries and values: replace with 0 all the token
        # positions between the true and the maximum sentence length.
        multiplication_mask = tf.tile(
            input=tf.expand_dims(tf.sequence_mask(sentence_lengths), axis=-1),
            multiples=[1, 1, num_units])  # [B, M, num_units]
        queries = tf.where(multiplication_mask, queries, tf.zeros_like(queries))
        keys = tf.where(multiplication_mask, keys, tf.zeros_like(keys))

        # Scaled dot-product attention.
        attention_evidence = tf.matmul(
            queries, tf.transpose(keys, [0, 2, 1]))  # [B, M, M]
        attention_evidence = tf.math.divide(
            attention_evidence, tf.constant(num_units ** 0.5))

        # Mask columns (with values of -infinity), based on rows that have 0 sum.
        attention_evidence_masked = mask(
            attention_evidence, queries, keys, mask_type="key")

        # Obtain the un-normalized attention weights.
        if attention_activation == "soft":
            attention_weights = tf.nn.sigmoid(attention_evidence_masked)
        elif attention_activation == "sharp":
            attention_weights = tf.exp(attention_evidence_masked)
        else:
            raise ValueError("Unknown/unsupported activation for attention: %s"
                             % attention_activation)
        attention_weights_unnormalized = attention_weights

        # Normalize attention weights.
        attention_weights /= tf.reduce_sum(
            attention_weights, axis=-1, keep_dims=True)  # [B, M, M]

        # Mask rows (with values of 0), based on columns that have 0 sum.
        attention_weights = mask(
            attention_weights, queries, keys, mask_type="query")
        attention_weights_unnormalized = mask(
            attention_weights_unnormalized, queries, keys, mask_type="query")

        # Obtain the token scores from the attention weights.
        # The token_scores below will have shape [B, M].
        if token_scoring_method == "sum":
            token_scores = tf.reduce_sum(
                attention_weights_unnormalized, axis=1)
        elif token_scoring_method == "max":
            token_scores = tf.reduce_max(
                attention_weights_unnormalized, axis=1)
        elif token_scoring_method == "avg":
            token_scores = tf.reduce_mean(
                attention_weights_unnormalized, axis=1)
        elif token_scoring_method == "logsumexp":
            token_scores = tf.reduce_logsumexp(
                attention_weights_unnormalized, axis=1)
        else:
            raise ValueError("Unknown/unsupported token scoring method: %s"
                             % token_scoring_method)

        token_scores_normalized = division_masking(
            inputs=token_scores, axis=-1,
            multiplies=[1, tf.shape(token_scores)[1]])  # [B, M]

        # Sentence scores as a weighted sum between the inputs and the attention weights.
        # weighted_sum_representation = tf.matmul(attention_weights, inputs)
        weighted_sum_representation = inputs * tf.expand_dims(
            token_scores_normalized, axis=-1)  # [B, M, num_units]

        processed_tensor = tf.reduce_sum(
            weighted_sum_representation, axis=1)  # [B, num_units]
        sentence_scores = tf.layers.dense(
            inputs=processed_tensor, units=1,
            activation=tf.sigmoid, kernel_initializer=initializer,
            name="sentence_scores_from_scaled_dot_product_ff")  # [B, 1]
        sentence_scores = tf.squeeze(sentence_scores, axis=-1)  # [B]
        return sentence_scores, token_scores, attention_weights


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
        separate_heads=True):
    """
    Computes token and sentence scores using a single-head attention mechanism,
    which can either be additive (mainly inspired by the single-head binary-label
    method above, as in Rei and Sogaard paper https://arxiv.org/pdf/1811.05949.pdf)
    or a scaled-dot product version (inspired by the transformer, but with just one head).
    Then, use these scores to obtain predictions at both granularities.
    :param inputs: 3D floats of shape [B, M, E]
    :param initializer: type of initializer (best if Glorot or Xavier)
    :param attention_activation
    :param num_sentence_labels: number of unique sentence labels
    :param num_heads: number of unique token labels
    :param sentence_lengths: the true sentence lengths, used for masking
    :param token_scoring_method
    :param scoring_activation: activation used for scoring, default is None.
    :param how_to_compute_attention: compute attention in the classic way (Marek) or as in transformer
    :param separate_heads: boolean value; when set to False, all heads
    are used to obtain the sentence scores; when set to True, the default and non-default heads
    from the token scores are used to obtain the sentence scores.
    :return sentence_scores: 2D floats of shape [B, num_sentence_labels]
    :return sentence_predictions: predicted labels for each sentence in the batch; ints of shape [B]
    :return token_scores: 3D floats of shape [B, M, num_heads]
    :return token_predictions: predicted labels for each token in each sentence; ints of shape [B, M]
    """
    with tf.variable_scope("transformer_single_heads_multi_attention"):
        token_scores_per_head = []
        sentence_scores_per_head = []
        attention_weights_per_head = []
        for i in range(num_heads):
            with tf.variable_scope("num_head_{}".format(i), reuse=tf.AUTO_REUSE):
                if how_to_compute_attention == "additive":
                    sentence_scores_head_i, token_scores_head_i, attention_weights_head_i = \
                        compute_scores_from_additive_attention(
                            inputs=inputs, initializer=initializer,
                            attention_activation=attention_activation,
                            sentence_lengths=sentence_lengths)
                elif how_to_compute_attention == "dot":
                    sentence_scores_head_i, token_scores_head_i, attention_weights_head_i = \
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
                attention_weights_per_head.append(attention_weights_head_i)

        sentence_scores = tf.stack(sentence_scores_per_head, axis=-1)  # [B, num_heads]

        if separate_heads:
            sentence_default_score = tf.layers.dense(
                inputs=tf.expand_dims(sentence_scores[:, 0], axis=-1), units=1,
                activation=scoring_activation, kernel_initializer=initializer,
                name="ff_non_default_sentence_scores")
            sentence_non_default_scores = tf.layers.dense(
                inputs=sentence_scores[:, 1:], units=num_sentence_labels-1,
                activation=scoring_activation, kernel_initializer=initializer,
                name="ff_default_sentence_scores")
            sentence_scores = tf.concat(
                [sentence_default_score, sentence_non_default_scores],
                axis=-1, name="sentence_scores_concatenation")
        else:
            sentence_scores = tf.layers.dense(
                inputs=sentence_scores, units=num_sentence_labels,
                activation=scoring_activation, kernel_initializer=initializer,
                name="ff_sentence_scores")  # [B, num_sentence_labels]

        sentence_probabilities = tf.nn.softmax(sentence_scores)
        sentence_predictions = tf.argmax(sentence_probabilities, axis=1)  # [B]

        token_scores = tf.stack(token_scores_per_head, axis=-1)  # [B, M, num_heads]
        token_probabilities = tf.nn.softmax(token_scores, axis=-1)  # [B, M, num_heads]
        token_predictions = tf.argmax(token_probabilities, axis=-1)  # [B, M]

        # Will be of shape [B, M, H] if an additive attention was used, or
        # of shape [B, M, M, H] if a scaled-dot product attention was used.
        attention_weights = tf.stack(attention_weights_per_head, axis=-1)

        return sentence_scores, sentence_predictions, token_scores, token_predictions, \
            token_probabilities, sentence_probabilities, attention_weights


def variant_1(
        inputs,
        initializer,
        attention_activation,
        num_sentence_labels,
        num_heads,
        hidden_units,
        sentence_lengths,
        scoring_activation=None,
        token_scoring_method="max",
        use_inputs_instead_values=False,
        separate_heads=True):
    """
    Variant 1 of the multi-head attention to obtain sentence and token scores and predictions.
    """
    with tf.variable_scope("variant_1"):
        num_units = inputs.get_shape().as_list()[-1]
        if num_units % num_heads != 0:
            num_units = ceil(num_units / num_heads) * num_heads
            inputs = tf.layers.dense(inputs, num_units)  # [B, M, num_units]

        # Project to get the queries, keys, and values.
        queries = tf.layers.dense(
            inputs, num_units, activation=tf.tanh,
            kernel_initializer=initializer)  # [B, M, num_units]
        keys = tf.layers.dense(
            inputs, num_units, activation=tf.tanh,
            kernel_initializer=initializer)  # [B, M, num_units]
        values = tf.layers.dense(
            inputs, num_units, activation=tf.tanh,
            kernel_initializer=initializer)  # [B, M, num_units]

        # Mask out the keys, queries and values: replace with 0 all the token
        # positions between the true and the maximum sentence length.
        multiplication_mask = tf.tile(
            input=tf.expand_dims(tf.sequence_mask(sentence_lengths), axis=-1),
            multiples=[1, 1, num_units])  # [B, M, num_units]
        queries = tf.where(multiplication_mask, queries, tf.zeros_like(queries))
        keys = tf.where(multiplication_mask, keys, tf.zeros_like(keys))

        # Split and concat as many projections as the number of heads.
        queries = tf.concat(
            tf.split(queries, num_heads, axis=2),
            axis=0)  # [B*num_heads, M, num_units/num_heads]
        keys = tf.concat(
            tf.split(keys, num_heads, axis=2),
            axis=0)  # [B*num_heads, M, num_units/num_heads]
        values = tf.concat(
            tf.split(values, num_heads, axis=2),
            axis=0)  # [B*num_heads, M, num_units/num_heads]
        inputs = tf.concat(
            tf.split(inputs, num_heads, axis=2),
            axis=0)  # [B*num_heads, M, num_units/num_heads]

        # Transpose multiplication and scale
        attention_evidence = tf.matmul(
            queries, tf.transpose(keys, [0, 2, 1]))  # [B*num_heads, M, M]
        attention_evidence = tf.math.divide(
            attention_evidence, tf.constant(num_units ** 0.5))

        # Mask columns (with values of -infinity), based on rows that have 0 sum.
        attention_evidence_masked = mask(
            attention_evidence, queries, keys, mask_type="key")

        # Apply a non-linear layer to obtain (un-normalized) attention weights.
        if attention_activation == "soft":
            attention_weights = tf.nn.sigmoid(attention_evidence_masked)
        elif attention_activation == "sharp":
            attention_weights = tf.math.exp(attention_evidence_masked)
        elif attention_activation == "linear":
            attention_weights = attention_evidence_masked
        else:
            raise ValueError("Unknown/unsupported attention activation: %s."
                             % attention_activation)

        attention_weights_unnormalized = attention_weights

        # Normalize attention weights.
        attention_weights /= tf.reduce_sum(
            attention_weights, axis=-1, keep_dims=True)

        # Mask rows (with values of 0), based on columns that have 0 sum.
        attention_weights = mask(
            attention_weights, queries, keys, mask_type="query")
        attention_weights_unnormalized = mask(
            attention_weights_unnormalized, queries, keys, mask_type="query")

        # [B*num_heads, M, num_units/num_heads]
        if use_inputs_instead_values:
            product = tf.matmul(attention_weights, inputs)
        else:
            product = tf.matmul(attention_weights, values)

        product = tf.reduce_sum(product, axis=1)  # [B*num_heads, num_units/num_heads]

        product = tf.layers.dense(
            inputs=product, units=hidden_units,
            activation=tf.tanh, kernel_initializer=initializer)  # [B*num_heads, hidden_units]

        processed_tensor = tf.layers.dense(
            inputs=product, units=1,
            kernel_initializer=initializer)  # [B*num_heads, 1]

        processed_tensor = tf.concat(
            tf.split(processed_tensor, num_heads), axis=1)  # [B, num_heads]

        if separate_heads:
            if num_sentence_labels == num_heads:
                sentence_scores = processed_tensor
            else:
                # Get the sentence representations corresponding to the default head.
                default_head = tf.gather(
                    processed_tensor,
                    indices=[0], axis=-1)  # [B, 1]

                # Get the sentence representations corresponding to the non-default head.
                non_default_heads = tf.gather(
                    processed_tensor,
                    indices=list(range(1, num_heads)), axis=-1)  # [B, num_heads-1]

                # Project onto one unit, corresponding to the default sentence label score.
                sentence_default_scores = tf.layers.dense(
                    default_head, units=1,
                    activation=scoring_activation, kernel_initializer=initializer,
                    name="sentence_default_scores_ff")  # [B, 1]

                # Project onto (num_sentence_labels-1) units, corresponding to
                # the non-default sentence label scores.
                sentence_non_default_scores = tf.layers.dense(
                    non_default_heads, units=num_sentence_labels - 1,
                    activation=scoring_activation, kernel_initializer=initializer,
                    name="sentence_non_default_scores_ff")  # [B, num_sentence_labels-1]

                sentence_scores = tf.concat(
                    [sentence_default_scores, sentence_non_default_scores],
                    axis=-1, name="sentence_scores_concatenation")  # [B, num_sent_labels]
        else:
            sentence_scores = tf.layers.dense(
                inputs=processed_tensor, units=num_sentence_labels,
                activation=scoring_activation, kernel_initializer=initializer,
                name="output_sent_specified_scores_ff")  # [B, num_sent_labels]

        sentence_probabilities = tf.nn.softmax(sentence_scores)
        sentence_predictions = tf.argmax(sentence_probabilities, axis=1)  # [B]

        # Obtain token scores from attention weights. Shape is [B*num_heads, M].
        if token_scoring_method == "sum":
            token_scores = tf.reduce_sum(attention_weights_unnormalized, axis=1)
        elif token_scoring_method == "max":
            token_scores = tf.reduce_max(attention_weights_unnormalized, axis=1)
        elif token_scoring_method == "avg":
            token_scores = tf.reduce_mean(attention_weights_unnormalized, axis=1)
        elif token_scoring_method == "logsumexp":
            token_scores = tf.reduce_logsumexp(attention_weights_unnormalized, axis=1)
        else:
            raise ValueError("Unknown/unsupported token scoring method: %s"
                             % token_scoring_method)

        token_scores = tf.expand_dims(token_scores, axis=2)  # [B*num_heads, M, 1]
        token_scores = tf.concat(
            tf.split(token_scores, num_heads), axis=2)  # [B, M, num_heads]
        token_probabilities = tf.nn.softmax(token_scores)
        token_predictions = tf.argmax(
            token_probabilities, axis=2, output_type=tf.int32)  # [B, M]

        attention_weights = tf.concat(
            tf.split(tf.expand_dims(attention_weights, axis=-1), num_heads),
            axis=-1)  # [B, M, M, num_heads]

        return sentence_scores, sentence_predictions, \
            token_scores, token_predictions, \
            token_probabilities, sentence_probabilities, attention_weights


def variant_2(
        inputs,
        initializer,
        attention_activation,
        num_sentence_labels,
        num_heads,
        hidden_units,
        sentence_lengths,
        scoring_activation=None,
        use_inputs_instead_values=False,
        separate_heads=True):
    """
    Variant 2 of the multi-head attention to obtain sentence and token scores and predictions.
    """
    with tf.variable_scope("variant_2"):
        num_units = inputs.get_shape().as_list()[-1]
        if num_units % num_heads != 0:
            num_units = ceil(num_units / num_heads) * num_heads
            inputs = tf.layers.dense(inputs, num_units)  # [B, M, num_units]

        # Project to get the queries, keys, and values.
        queries = tf.layers.dense(
            inputs, num_units, activation=tf.tanh,
            kernel_initializer=initializer)  # [B, M, num_units]
        keys = tf.layers.dense(
            inputs, num_units, activation=tf.tanh,
            kernel_initializer=initializer)  # [B, M, num_units]
        values = tf.layers.dense(
            inputs, num_units, activation=tf.tanh,
            kernel_initializer=initializer)  # [B, M, num_units]

        # Mask out the keys, queries and values: replace with 0 all the token
        # positions between the true and the maximum sentence length.
        multiplication_mask = tf.tile(
            input=tf.expand_dims(tf.sequence_mask(sentence_lengths), axis=-1),
            multiples=[1, 1, num_units])  # [B, M, num_units]
        keys = tf.where(multiplication_mask, keys, tf.zeros_like(keys))

        # Split and concat as many projections as the number of heads.
        queries = tf.concat(
            tf.split(queries, num_heads, axis=2),
            axis=0)  # [B*num_heads, M, num_units/num_heads]

        # [B*num_heads, 1, num_units/num_heads]
        queries = tf.reduce_sum(queries, axis=1, keep_dims=True)

        keys = tf.concat(
            tf.split(keys, num_heads, axis=2),
            axis=0)  # [B*num_heads, M, num_units/num_heads]
        values = tf.concat(
            tf.split(values, num_heads, axis=2),
            axis=0)  # [B*num_heads, M, num_units/num_heads]
        inputs = tf.concat(
            tf.split(inputs, num_heads, axis=2),
            axis=0)  # [B*num_heads, M, num_units/num_heads]

        # Transpose multiplication and scale
        attention_evidence = tf.matmul(
            queries, tf.transpose(keys, [0, 2, 1]))  # [B*num_heads, 1, M]
        attention_evidence = tf.math.divide(
            attention_evidence, tf.constant(num_units ** 0.5))

        # Mask columns (with values of -infinity), based on rows that have 0 sum.
        attention_evidence_masked = mask(
            attention_evidence, queries, keys, mask_type="key")

        # Apply a non-linear layer to obtain (un-normalized) attention weights.
        if attention_activation == "soft":
            attention_weights = tf.nn.sigmoid(attention_evidence_masked)
        elif attention_activation == "sharp":
            attention_weights = tf.math.exp(attention_evidence_masked)
        elif attention_activation == "linear":
            attention_weights = attention_evidence_masked
        else:
            raise ValueError("Unknown/unsupported attention activation: %s."
                             % attention_activation)

        attention_weights_unnormalized = attention_weights

        # Normalize attention weights.
        attention_weights /= tf.reduce_sum(
            attention_weights, axis=-1, keep_dims=True)

        # Mask rows (with values of 0), based on columns that have 0 sum.
        attention_weights = mask(
            attention_weights, queries, keys, mask_type="query")
        attention_weights_unnormalized = mask(
            attention_weights_unnormalized, queries, keys, mask_type="query")

        # Transpose attention weights.
        attention_weights = tf.transpose(
            attention_weights, [0, 2, 1])  # [B*num_heads, M, 1]

        # [B*num_heads, M, num_units/num_heads]
        if use_inputs_instead_values:
            product = inputs * attention_weights
        else:
            product = values * attention_weights

        product = tf.reduce_sum(product, axis=1)  # [B*num_heads, num_units/num_heads]

        product = tf.layers.dense(
            inputs=product, units=hidden_units,
            activation=tf.tanh, kernel_initializer=initializer)  # [B*num_heads, hidden_units]

        processed_tensor = tf.layers.dense(
            inputs=product, units=1,
            kernel_initializer=initializer)  # [B*num_heads, 1]

        processed_tensor = tf.concat(
            tf.split(processed_tensor, num_heads), axis=1)  # [B, num_heads]

        if separate_heads:
            if num_sentence_labels == num_heads:
                sentence_scores = processed_tensor
            else:
                # Get the sentence representations corresponding to the default head.
                default_head = tf.gather(
                    processed_tensor,
                    indices=[0], axis=-1)  # [B, 1]

                # Get the sentence representations corresponding to the non-default head.
                non_default_heads = tf.gather(
                    processed_tensor,
                    indices=list(range(1, num_heads)), axis=-1)  # [B, num_heads-1]

                # Project onto one unit, corresponding to the default sentence label score.
                sentence_default_scores = tf.layers.dense(
                    default_head, units=1,
                    activation=scoring_activation, kernel_initializer=initializer,
                    name="sentence_default_scores_ff")  # [B, 1]

                # Project onto (num_sentence_labels-1) units, corresponding to
                # the non-default sentence label scores.
                sentence_non_default_scores = tf.layers.dense(
                    non_default_heads, units=num_sentence_labels - 1,
                    activation=scoring_activation, kernel_initializer=initializer,
                    name="sentence_non_default_scores_ff")  # [B, num_sentence_labels-1]

                sentence_scores = tf.concat(
                    [sentence_default_scores, sentence_non_default_scores],
                    axis=-1, name="sentence_scores_concatenation")  # [B, num_sent_labels]
        else:
            sentence_scores = tf.layers.dense(
                inputs=processed_tensor, units=num_sentence_labels,
                activation=scoring_activation, kernel_initializer=initializer,
                name="output_sent_specified_scores_ff")  # [B, num_sent_labels]

        sentence_probabilities = tf.nn.softmax(sentence_scores)
        sentence_predictions = tf.argmax(sentence_probabilities, axis=1)  # [B]

        # Obtain token scores from attention weights.
        token_scores = tf.transpose(
            attention_weights_unnormalized, [0, 2, 1])  # [num_heads*B, M, 1]
        token_scores = tf.concat(
            tf.split(token_scores, num_heads), axis=2)  # [B, M, num_heads]
        token_probabilities = tf.nn.softmax(token_scores)
        token_predictions = tf.argmax(
            token_probabilities, axis=2, output_type=tf.int32)  # [B, M]

        attention_weights = tf.concat(
            tf.split(tf.transpose(attention_weights, [0, 2, 1]), num_heads),
            axis=-1)  # [B, M, num_heads]

        return sentence_scores, sentence_predictions, \
            token_scores, token_predictions, \
            token_probabilities, sentence_probabilities, attention_weights


def variant_3(
        inputs,
        initializer,
        attention_activation,
        num_sentence_labels,
        num_heads,
        attention_size,
        sentence_lengths,
        scoring_activation=None,
        separate_heads=True):
    """
    Variant 3 of the multi-head attention to obtain sentence and token scores and predictions.
    """
    with tf.variable_scope("variant_3"):
        num_units = inputs.get_shape().as_list()[-1]
        if num_units % num_heads != 0:
            num_units = ceil(num_units / num_heads) * num_heads
            inputs = tf.layers.dense(inputs, num_units)  # [B, M, num_units]

        # Trainable parameters
        w_omega = tf.Variable(
            tf.random_normal([num_heads, num_units, attention_size],
                             stddev=0.1))  # [num_heads, num_units, A]
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        # Computing the attention score, of shape [B, M, H, A].
        attention_evidence = tf.tanh(tf.tensordot(inputs, w_omega, axes=[[2], [1]]) + b_omega)
        attention_evidence = tf.tensordot(
            attention_evidence, u_omega, axes=[[-1], [0]],
            name='attention_evidence_score')  # [B, M, H]

        # Apply a non-linear layer to obtain (un-normalized) attention weights.
        if attention_activation == "soft":
            attention_weights_unnormalized = tf.nn.sigmoid(attention_evidence)
        elif attention_activation == "sharp":
            attention_weights_unnormalized = tf.math.exp(attention_evidence)
        elif attention_activation == "linear":
            attention_weights_unnormalized = attention_evidence
        else:
            raise ValueError("Unknown/unsupported attention activation: %s."
                             % attention_activation)

        tiled_sentence_lengths = tf.tile(
            input=tf.expand_dims(
                tf.sequence_mask(sentence_lengths), axis=-1),
            multiples=[1, 1, num_heads])

        attention_weights_unnormalized = tf.where(
            tiled_sentence_lengths,
            attention_weights_unnormalized,
            tf.zeros_like(attention_weights_unnormalized))

        attention_weights = attention_weights_unnormalized / tf.reduce_sum(
            attention_weights_unnormalized, axis=1, keep_dims=True)  # [B, M, H]

        # Prepare alphas and input.
        attention_weights = tf.transpose(attention_weights, [0, 2, 1])  # [B, H, M, 1]
        inputs = tf.tile(
            input=tf.expand_dims(inputs, axis=1),
            multiples=[1, num_heads, 1, 1])  # [B, H, M, E]

        product = inputs * tf.expand_dims(attention_weights, axis=-1)  # [B, H, M, E]
        output = tf.reduce_sum(product, axis=2)  # [B, H, E]

        processed_tensor = tf.squeeze(tf.layers.dense(
            inputs=output, units=1,
            kernel_initializer=initializer), axis=-1)  # [B, num_heads]

        if separate_heads:
            if num_sentence_labels == num_heads:
                sentence_scores = processed_tensor
            else:
                # Get the sentence representations corresponding to the default head.
                default_head = tf.gather(
                    processed_tensor,
                    indices=[0], axis=-1)  # [B, 1]

                # Get the sentence representations corresponding to the non-default head.
                non_default_heads = tf.gather(
                    processed_tensor,
                    indices=list(range(1, num_heads)), axis=-1)  # [B, num_heads-1]

                # Project onto one unit, corresponding to the default sentence label score.
                sentence_default_scores = tf.layers.dense(
                    default_head, units=1,
                    activation=scoring_activation, kernel_initializer=initializer,
                    name="sentence_default_scores_ff")  # [B, 1]

                # Project onto (num_sentence_labels-1) units, corresponding to
                # the non-default sentence label scores.
                sentence_non_default_scores = tf.layers.dense(
                    non_default_heads, units=num_sentence_labels - 1,
                    activation=scoring_activation, kernel_initializer=initializer,
                    name="sentence_non_default_scores_ff")  # [B, num_sentence_labels-1]

                sentence_scores = tf.concat(
                    [sentence_default_scores, sentence_non_default_scores],
                    axis=-1, name="sentence_scores_concatenation")  # [B, num_sent_labels]
        else:
            sentence_scores = tf.layers.dense(
                inputs=processed_tensor, units=num_sentence_labels,
                activation=scoring_activation, kernel_initializer=initializer,
                name="output_sent_specified_scores_ff")  # [B, num_sent_labels]

        sentence_probabilities = tf.nn.softmax(sentence_scores)
        sentence_predictions = tf.argmax(sentence_probabilities, axis=1)  # [B]

        token_scores = attention_weights_unnormalized  # [B, M, num_heads]
        token_probabilities = tf.nn.softmax(token_scores)
        token_predictions = tf.argmax(
            token_probabilities, axis=2, output_type=tf.int32)  # [B, M]

        return sentence_scores, sentence_predictions, \
            token_scores, token_predictions, \
            token_probabilities, sentence_probabilities, attention_weights


def variant_4(
        inputs,
        initializer,
        attention_activation,
        num_sentence_labels,
        num_heads,
        hidden_units,
        sentence_lengths,
        scoring_activation=None,
        token_scoring_method="max",
        use_inputs_instead_values=False,
        separate_heads=True):
    """
    Variant 4 of the multi-head attention to obtain sentence and token scores and predictions.
    """
    with tf.variable_scope("variant_4"):
        num_units = inputs.get_shape().as_list()[-1]
        if num_units % num_heads != 0:
            num_units = ceil(num_units / num_heads) * num_heads
            inputs = tf.layers.dense(inputs, num_units)  # [B, M, num_units]

        # Project to get the queries, keys, and values.
        queries = tf.layers.dense(
            inputs, num_units, activation=tf.tanh,
            kernel_initializer=initializer)  # [B, M, num_units]
        keys = tf.layers.dense(
            inputs, num_units, activation=tf.tanh,
            kernel_initializer=initializer)  # [B, M, num_units]
        values = tf.layers.dense(
            inputs, num_units, activation=tf.tanh,
            kernel_initializer=initializer)  # [B, M, num_units]

        # Mask out the keys, queries and values: replace with 0 all the token
        # positions between the true and the maximum sentence length.
        multiplication_mask = tf.tile(
            input=tf.expand_dims(tf.sequence_mask(sentence_lengths), axis=-1),
            multiples=[1, 1, num_units])  # [B, M, num_units]
        queries = tf.where(multiplication_mask, queries, tf.zeros_like(queries))
        keys = tf.where(multiplication_mask, keys, tf.zeros_like(keys))
        values = tf.where(multiplication_mask, values, tf.zeros_like(values))

        # Split and concat as many projections as the number of heads.
        queries = tf.concat(
            tf.split(queries, num_heads, axis=2),
            axis=0)  # [B*num_heads, M, num_units/num_heads]
        keys = tf.concat(
            tf.split(keys, num_heads, axis=2),
            axis=0)  # [B*num_heads, M, num_units/num_heads]
        values = tf.concat(
            tf.split(values, num_heads, axis=2),
            axis=0)  # [B*num_heads, M, num_units/num_heads]
        inputs = tf.concat(
            tf.split(inputs, num_heads, axis=2),
            axis=0)  # [B*num_heads, M, num_units/num_heads]

        # Transpose multiplication and scale
        attention_evidence = tf.matmul(
            queries, tf.transpose(keys, [0, 2, 1]))  # [B*num_heads, M, M]
        attention_evidence = tf.math.divide(
            attention_evidence, tf.constant(num_units ** 0.5))

        # Mask columns (with values of -infinity), based on rows that have 0 sum.
        attention_evidence_masked = mask(
            attention_evidence, queries, keys, mask_type="key")

        # Apply a non-linear layer to obtain (un-normalized) attention weights.
        if attention_activation == "soft":
            attention_weights_unnormalized = tf.nn.sigmoid(attention_evidence_masked)
        elif attention_activation == "sharp":
            attention_weights_unnormalized = tf.math.exp(attention_evidence_masked)
        elif attention_activation == "linear":
            attention_weights_unnormalized = attention_evidence_masked
        else:
            raise ValueError("Unknown/unsupported attention activation: %s."
                             % attention_activation)

        attention_weights_unnormalized = mask(  # [B*num_heads, M, M]
            attention_weights_unnormalized, queries, keys, mask_type="query")

        # Obtain token scores from attention weights. Shape is [B*num_heads, M].
        if token_scoring_method == "sum":
            attention_weights_unnormalized = tf.reduce_sum(
                attention_weights_unnormalized, axis=1)
        elif token_scoring_method == "max":
            attention_weights_unnormalized = tf.reduce_max(
                attention_weights_unnormalized, axis=1)
        elif token_scoring_method == "avg":
            attention_weights_unnormalized = tf.reduce_mean(
                attention_weights_unnormalized, axis=1)
        elif token_scoring_method == "logsumexp":
            attention_weights_unnormalized = tf.reduce_logsumexp(
                attention_weights_unnormalized, axis=1)
        else:
            raise ValueError("Unknown/unsupported token scoring method: %s"
                             % token_scoring_method)

        # Normalize to obtain attention weights.
        attention_weights = attention_weights_unnormalized / tf.reduce_sum(
            attention_weights_unnormalized, axis=1, keep_dims=True)

        token_scores = tf.concat(
            tf.split(tf.expand_dims(attention_weights_unnormalized, axis=2), num_heads),
            axis=2)  # [B, M, num_heads]
        token_probabilities = tf.nn.softmax(token_scores)
        token_predictions = tf.argmax(
            token_probabilities, axis=2, output_type=tf.int32)  # [B, M]

        if use_inputs_instead_values:
            product = tf.reduce_sum(inputs * tf.expand_dims(attention_weights, axis=-1),
                                    axis=1)  # [B*num_heads, num_units/num_heads]
        else:
            product = tf.reduce_sum(values * tf.expand_dims(attention_weights, axis=-1),
                                    axis=1)  # [B*num_heads, num_units/num_heads]

        product = tf.layers.dense(
            inputs=product, units=hidden_units,
            activation=tf.tanh, kernel_initializer=initializer)  # [B*num_heads, hidden_units]

        processed_tensor = tf.layers.dense(
            inputs=product, units=1,
            kernel_initializer=initializer)  # [B*num_heads, 1]

        processed_tensor = tf.concat(
            tf.split(processed_tensor, num_heads), axis=1)  # [B, num_heads]

        if separate_heads:
            if num_sentence_labels == num_heads:
                sentence_scores = processed_tensor
            else:
                # Get the sentence representations corresponding to the default head.
                default_head = tf.gather(
                    processed_tensor,
                    indices=[0], axis=-1)  # [B, 1]

                # Get the sentence representations corresponding to the non-default head.
                non_default_heads = tf.gather(
                    processed_tensor,
                    indices=list(range(1, num_heads)), axis=-1)  # [B, num_heads-1]

                # Project onto one unit, corresponding to the default sentence label score.
                sentence_default_scores = tf.layers.dense(
                    default_head, units=1,
                    activation=scoring_activation, kernel_initializer=initializer,
                    name="sentence_default_scores_ff")  # [B, 1]

                # Project onto (num_sentence_labels-1) units, corresponding to
                # the non-default sentence label scores.
                sentence_non_default_scores = tf.layers.dense(
                    non_default_heads, units=num_sentence_labels - 1,
                    activation=scoring_activation, kernel_initializer=initializer,
                    name="sentence_non_default_scores_ff")  # [B, num_sentence_labels-1]

                sentence_scores = tf.concat(
                    [sentence_default_scores, sentence_non_default_scores],
                    axis=-1, name="sentence_scores_concatenation")  # [B, num_sent_labels]
        else:
            sentence_scores = tf.layers.dense(
                inputs=processed_tensor, units=num_sentence_labels,
                activation=scoring_activation, kernel_initializer=initializer,
                name="output_sent_specified_scores_ff")  # [B, num_sent_labels]

        sentence_probabilities = tf.nn.softmax(sentence_scores)
        sentence_predictions = tf.argmax(sentence_probabilities, axis=1)  # [B]

        attention_weights = tf.concat(
            tf.split(tf.expand_dims(attention_weights, axis=-1), num_heads),
            axis=-1)  # [B, M, num_heads]

        return sentence_scores, sentence_predictions, \
            token_scores, token_predictions, \
            token_probabilities, sentence_probabilities, attention_weights


def variant_5(
        inputs,
        initializer,
        attention_activation,
        num_sentence_labels,
        num_heads,
        hidden_units,
        sentence_lengths,
        scoring_activation=None,
        token_scoring_method="max",
        use_inputs_instead_values=False,
        separate_heads=True):
    """
    Variant 5 of the multi-head attention to obtain sentence and token scores and predictions.
    """
    with tf.variable_scope("variant_5"):
        num_units = inputs.get_shape().as_list()[-1]
        if num_units % num_heads != 0:
            num_units = ceil(num_units / num_heads) * num_heads
            inputs = tf.layers.dense(inputs, num_units)  # [B, M, num_units]

        # Project to get the queries, keys, and values.
        queries = tf.layers.dense(
            inputs, num_units, activation=tf.tanh,
            kernel_initializer=initializer)  # [B, M, num_units]
        keys = tf.layers.dense(
            inputs, num_units, activation=tf.tanh,
            kernel_initializer=initializer)  # [B, M, num_units]
        values = tf.layers.dense(
            inputs, num_units, activation=tf.tanh,
            kernel_initializer=initializer)  # [B, M, num_units]

        # Mask out the keys, queries and values: replace with 0 all the token
        # positions between the true and the maximum sentence length.
        multiplication_mask = tf.tile(
            input=tf.expand_dims(tf.sequence_mask(sentence_lengths), axis=-1),
            multiples=[1, 1, num_units])  # [B, M, num_units]
        queries = tf.where(multiplication_mask, queries, tf.zeros_like(queries))
        keys = tf.where(multiplication_mask, keys, tf.zeros_like(keys))
        values = tf.where(multiplication_mask, values, tf.zeros_like(values))

        # Split and concat as many projections as the number of heads.
        queries = tf.concat(
            tf.split(queries, num_heads, axis=2),
            axis=0)  # [B*num_heads, M, num_units/num_heads]
        keys = tf.concat(
            tf.split(keys, num_heads, axis=2),
            axis=0)  # [B*num_heads, M, num_units/num_heads]
        values = tf.concat(
            tf.split(values, num_heads, axis=2),
            axis=0)  # [B*num_heads, M, num_units/num_heads]
        inputs = tf.concat(
            tf.split(inputs, num_heads, axis=2),
            axis=0)  # [B*num_heads, M, num_units/num_heads]

        # Transpose multiplication and scale
        attention_evidence = tf.matmul(
            queries, tf.transpose(keys, [0, 2, 1]))  # [B*num_heads, M, M]
        attention_evidence = tf.math.divide(
            attention_evidence, tf.constant(num_units ** 0.5))

        # Obtain token scores from attention weights. Shape is [B*num_heads, M].
        if token_scoring_method == "sum":
            attention_evidence = tf.reduce_sum(
                attention_evidence, axis=1)
        elif token_scoring_method == "max":
            attention_evidence = tf.reduce_max(
                attention_evidence, axis=1)
        elif token_scoring_method == "avg":
            attention_evidence = tf.reduce_mean(
                attention_evidence, axis=1)
        elif token_scoring_method == "logsumexp":
            attention_evidence = tf.reduce_logsumexp(
                attention_evidence, axis=1)
        else:
            raise ValueError("Unknown/unsupported token scoring method: %s"
                             % token_scoring_method)

        # Apply a non-linear layer to obtain un-normalized attention weights.
        if attention_activation == "soft":
            attention_weights_unnormalized = tf.nn.sigmoid(attention_evidence)
        elif attention_activation == "sharp":
            attention_weights_unnormalized = tf.math.exp(attention_evidence)
        elif attention_activation == "linear":
            attention_weights_unnormalized = attention_evidence
        else:
            raise ValueError("Unknown/unsupported attention activation: %s."
                             % attention_activation)

        tiled_sentence_lengths = tf.tile(
            input=tf.sequence_mask(sentence_lengths), multiples=[num_heads, 1])

        attention_weights_unnormalized = tf.where(
             tiled_sentence_lengths,
             attention_weights_unnormalized,
             tf.zeros_like(attention_weights_unnormalized))

        # Normalize to obtain attention weights of shape [B*num_heads, M].
        attention_weights = attention_weights_unnormalized / tf.reduce_sum(
            attention_weights_unnormalized, axis=1, keep_dims=True)

        token_scores = tf.concat(
            tf.split(tf.expand_dims(attention_weights_unnormalized, axis=2), num_heads),
            axis=2)  # [B, M, num_heads]
        token_probabilities = tf.nn.softmax(token_scores)
        token_predictions = tf.argmax(
            token_probabilities, axis=2, output_type=tf.int32)  # [B, M]

        if use_inputs_instead_values:
            product = tf.reduce_sum(inputs * tf.expand_dims(attention_weights, axis=-1),
                                    axis=1)  # [B*num_heads, num_units/num_heads]
        else:
            product = tf.reduce_sum(values * tf.expand_dims(attention_weights, axis=-1),
                                    axis=1)  # [B*num_heads, num_units/num_heads]

        product = tf.layers.dense(
            inputs=product, units=hidden_units,
            activation=tf.tanh, kernel_initializer=initializer)  # [B*num_heads, hidden_units]

        processed_tensor = tf.layers.dense(
            inputs=product, units=1,
            kernel_initializer=initializer)  # [B*num_heads, 1]

        processed_tensor = tf.concat(
            tf.split(processed_tensor, num_heads), axis=1)  # [B, num_heads]

        if separate_heads:
            if num_sentence_labels == num_heads:
                sentence_scores = processed_tensor
            else:
                # Get the sentence representations corresponding to the default head.
                default_head = tf.gather(
                    processed_tensor,
                    indices=[0], axis=-1)  # [B, 1]

                # Get the sentence representations corresponding to the non-default head.
                non_default_heads = tf.gather(
                    processed_tensor,
                    indices=list(range(1, num_heads)), axis=-1)  # [B, num_heads-1]

                # Project onto one unit, corresponding to the default sentence label score.
                sentence_default_scores = tf.layers.dense(
                    default_head, units=1,
                    activation=scoring_activation, kernel_initializer=initializer,
                    name="sentence_default_scores_ff")  # [B, 1]

                # Project onto (num_sentence_labels-1) units, corresponding to
                # the non-default sentence label scores.
                sentence_non_default_scores = tf.layers.dense(
                    non_default_heads, units=num_sentence_labels - 1,
                    activation=scoring_activation, kernel_initializer=initializer,
                    name="sentence_non_default_scores_ff")  # [B, num_sentence_labels-1]

                sentence_scores = tf.concat(
                    [sentence_default_scores, sentence_non_default_scores],
                    axis=-1, name="sentence_scores_concatenation")  # [B, num_sent_labels]
        else:
            sentence_scores = tf.layers.dense(
                inputs=processed_tensor, units=num_sentence_labels,
                activation=scoring_activation, kernel_initializer=initializer,
                name="output_sent_specified_scores_ff")  # [B, num_sent_labels]

        sentence_probabilities = tf.nn.softmax(sentence_scores)
        sentence_predictions = tf.argmax(sentence_probabilities, axis=1)  # [B]

        attention_weights = tf.concat(
            tf.split(tf.expand_dims(attention_weights, axis=-1), num_heads),
            axis=-1)  # [B, M, num_heads]

        return sentence_scores, sentence_predictions, \
            token_scores, token_predictions, \
            token_probabilities, sentence_probabilities, attention_weights


def variant_6(
        inputs,
        initializer,
        attention_activation,
        num_sentence_labels,
        num_heads,
        hidden_units,
        scoring_activation=None,
        token_scoring_method="max",
        separate_heads=True):
    """
    Variant 6 of the multi-head attention to obtain sentence and token scores and predictions.
    """
    with tf.variable_scope("variant_6"):
        num_units = inputs.get_shape().as_list()[-1]
        keys_list = []
        queries_list = []
        values_list = []

        for i in range(num_heads):
            with tf.variable_scope("num_head_{}".format(i), reuse=tf.AUTO_REUSE):
                keys_this_head = tf.layers.dense(
                    inputs, num_units, activation=tf.tanh,
                    kernel_initializer=initializer)  # [B, M, num_units]
                queries_this_head = tf.layers.dense(
                    inputs, num_units, activation=tf.nn.relu,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.7),
                    kernel_initializer=initializer)  # [B, M, num_units]
                values_this_head = tf.layers.dense(
                    inputs, num_units, activation=tf.tanh,
                    kernel_initializer=initializer)  # [B, M, num_units]

                keys_list.append(keys_this_head)
                queries_list.append(queries_this_head)
                values_list.append(values_this_head)

        keys = tf.stack(keys_list)  # [num_heads, B, M, num_units]
        queries = tf.stack(queries_list)   # [num_heads, B, M, num_units]
        values = tf.stack(values_list)  # [num_heads, B, M, num_units]

        # Transpose multiplication and scale
        attention_evidence = tf.matmul(
            queries, tf.transpose(keys, [0, 1, 3, 2]))  # [num_heads, B, M, M]
        attention_evidence = tf.math.divide(
            attention_evidence, tf.constant(num_units ** 0.5))

        # Mask columns (with values of -infinity), based on rows that have 0 sum.
        attention_evidence_masked = mask_2(
            attention_evidence, queries, keys, mask_type="key")

        # Apply a non-linear layer to obtain (un-normalized) attention weights.
        if attention_activation == "soft":
            attention_weights = tf.nn.sigmoid(attention_evidence_masked)
        elif attention_activation == "sharp":
            attention_weights = tf.math.exp(attention_evidence_masked)
        elif attention_activation == "linear":
            attention_weights = attention_evidence_masked
        else:
            raise ValueError("Unknown/unsupported attention activation: %s."
                             % attention_activation)

        attention_weights_unnormalized = attention_weights

        # Normalize attention weights.
        attention_weights /= tf.reduce_sum(
            attention_weights, axis=-1, keep_dims=True)

        # Mask rows (with values of 0), based on columns that have 0 sum.
        attention_weights = mask_2(
            attention_weights, queries, keys, mask_type="query")
        attention_weights_unnormalized = mask_2(
            attention_weights_unnormalized, queries, keys, mask_type="query")

        # [num_heads, B, M, num_units]
        product = tf.matmul(attention_weights, values)

        product = tf.reduce_sum(product, axis=2)  # [num_heads, B, num_units]

        product = tf.layers.dense(
            inputs=product, units=hidden_units,
            activation=tf.tanh, kernel_initializer=initializer)  # [num_heads, B, hidden_units]

        processed_tensor = tf.layers.dense(
            inputs=product, units=1,
            kernel_initializer=initializer)  # [num_heads, B, 1]

        processed_tensor = tf.transpose(
            tf.squeeze(processed_tensor, axis=-1), [1, 0])  # [B, num_heads]

        if separate_heads:
            if num_sentence_labels == num_heads:
                sentence_scores = processed_tensor
            else:
                # Get the sentence representations corresponding to the default head.
                default_head = tf.gather(
                    processed_tensor,
                    indices=[0], axis=-1)  # [B, 1]

                # Get the sentence representations corresponding to the non-default head.
                non_default_heads = tf.gather(
                    processed_tensor,
                    indices=list(range(1, num_heads)), axis=-1)  # [B, num_heads-1]

                # Project onto one unit, corresponding to the default sentence label score.
                sentence_default_scores = tf.layers.dense(
                    default_head, units=1,
                    activation=scoring_activation, kernel_initializer=initializer,
                    name="sentence_default_scores_ff")  # [B, 1]

                # Project onto (num_sentence_labels-1) units, corresponding to
                # the non-default sentence label scores.
                sentence_non_default_scores = tf.layers.dense(
                    non_default_heads, units=num_sentence_labels - 1,
                    activation=scoring_activation, kernel_initializer=initializer,
                    name="sentence_non_default_scores_ff")  # [B, num_sentence_labels-1]

                sentence_scores = tf.concat(
                    [sentence_default_scores, sentence_non_default_scores],
                    axis=-1, name="sentence_scores_concatenation")  # [B, num_sent_labels]
        else:
            sentence_scores = tf.layers.dense(
                inputs=processed_tensor, units=num_sentence_labels,
                activation=scoring_activation, kernel_initializer=initializer,
                name="output_sent_specified_scores_ff")  # [B, num_sent_labels]

        sentence_probabilities = tf.nn.softmax(sentence_scores)
        sentence_predictions = tf.argmax(sentence_probabilities, axis=1)  # [B]

        # Obtain token scores from attention weights. Shape is [num_heads, B, M].
        if token_scoring_method == "sum":
            token_scores = tf.reduce_sum(attention_weights_unnormalized, axis=2)
        elif token_scoring_method == "max":
            token_scores = tf.reduce_max(attention_weights_unnormalized, axis=2)
        elif token_scoring_method == "avg":
            token_scores = tf.reduce_mean(attention_weights_unnormalized, axis=2)
        elif token_scoring_method == "logsumexp":
            token_scores = tf.reduce_logsumexp(attention_weights_unnormalized, axis=2)
        else:
            raise ValueError("Unknown/unsupported token scoring method: %s"
                             % token_scoring_method)

        token_scores = tf.transpose(token_scores, [1, 2, 0])  # [B, M, num_heads]
        token_probabilities = tf.nn.softmax(token_scores)
        token_predictions = tf.argmax(
            token_probabilities, axis=2, output_type=tf.int32)  # [B, M]

        attention_weights = tf.transpose(attention_weights, [1, 2, 3, 0])  # [B, M, M, num_heads]

        return sentence_scores, sentence_predictions, \
            token_scores, token_predictions, \
            token_probabilities, sentence_probabilities, attention_weights


def get_token_representative_values(token_probabilities, approach):
    """
    Obtains the token probabilities representative for each head across the sentence.
    :param token_probabilities: the softmaxed token scores.
    :param approach: how to get the representations (max, avg, log).
    :return: token_representative_values of shape [batch_size, num_heads].
    """
    if "max" in approach:
        token_representative_values = tf.reduce_max(
            token_probabilities, axis=1)
    elif "avg" in approach:
        token_representative_values = tf.reduce_max(
            token_probabilities, axis=1)
    elif "log" in approach:
        token_representative_values = tf.reduce_logsumexp(
            token_probabilities, axis=1)
    else:
        raise ValueError("Unknown approach for getting "
                         "token representative values: %s." % approach)
    return token_representative_values  # [B, num_heads]


def get_one_hot_of_token_labels_length(
        sentence_labels, num_sent_labels, num_tok_labels):
    """
    Obtains one-hot sentence representations.
    :param sentence_labels: ground truth sentence labels.
    :param num_sent_labels: total number of unique sentence labels.
    :param num_tok_labels: total number of unique token labels.
    :return: one hot sentence labels, corresponding to the token labels.
    """
    one_hot_sentence_labels = tf.one_hot(
        tf.cast(sentence_labels, tf.int64),
        depth=num_sent_labels)

    if num_sent_labels == 2 and num_sent_labels != num_tok_labels:
        # Get the default and non-default sentence labels.
        default_sentence_labels = tf.gather(
            one_hot_sentence_labels, indices=[0], axis=-1)  # [B x 1]
        non_default_sentence_labels = tf.gather(
            one_hot_sentence_labels, indices=[1], axis=-1)  # [B x 1]

        # Tile the non-default one (num_tok_labels - 1) times.
        tiled_non_default_sentence_labels = tf.tile(
            input=non_default_sentence_labels,
            multiples=[1, num_tok_labels - 1])

        # Get one-hot sentence labels of shape [B, num_tok_labels].
        one_hot_sentence_labels = tf.concat(
            [default_sentence_labels, tiled_non_default_sentence_labels],
            axis=-1, name="one_hot_sentence_labels_concatenation")
    return one_hot_sentence_labels  # [B, num_tok_labels]


def compute_attention_loss(
        token_probabilities, sentence_labels,
        num_sent_labels, num_tok_labels,
        approach, compute_pairwise=False):
    """
    Attention-level loss -- currently, implementation possible only in two cases:
      1. The number of sentence labels is equal to the number of token labels.
         In this case, the attention loss is computed element-wise (for each label).
      2. The number of sentence labels is 2, while the number of tokens is arbitrary.
         In this case, two scores are computed from the token scores:
              * one corresponding to the default label
              * one corresponding to the rest of labels (non-default labels)
    :param token_probabilities: 3D tensor, shape [B, M, num_tok_labels]
                                that are normalized across heads (last axis).
    :param sentence_labels: 2D tensor, shape [B, num_labels_tok]
    :param num_sent_labels: number of unique sentence labels.
    :param num_tok_labels: number of unique token labels.
    :param approach: method to extract token representation values.
    :param compute_pairwise: whether to compute the loss pairwise or not.
    :return: a number representing the sum over attention losses computed.
    """
    if num_sent_labels == num_tok_labels or num_sent_labels == 2:
        # Compute the token representations based on the approach selected.
        token_representative_values = get_token_representative_values(
            token_probabilities, approach)  # [B, num_heads]

        one_hot_sentence_labels = get_one_hot_of_token_labels_length(
            sentence_labels, num_sent_labels, num_tok_labels)
        if compute_pairwise:
            attention_loss = tf.losses.mean_pairwise_squared_error(
                labels=label_smoothing(one_hot_sentence_labels, epsilon=0.15),
                predictions=token_representative_values, weights=1.15)
        else:
            attention_loss = tf.square(
                token_representative_values -
                label_smoothing(one_hot_sentence_labels, epsilon=0.15))
    else:
        raise ValueError(
            "You have different number of token labels (%d) and "
            "sentence labels (%d, which is non-binary). "
            "We don't support attention loss for such a case!"
            % (num_tok_labels, num_sent_labels))
    return attention_loss


def compute_gap_distance_loss(
        token_probabilities, sentence_labels,
        num_sent_labels, num_tok_labels,
        minimum_gap_distance, approach,
        type_distance):
    """
    Gap-distance loss: the intuition is that the gap between the default
    and non-default scores should be wider than a certain threshold.
    :param token_probabilities: 3D tensor, shape [B, M, num_tok_labels]
                                that are normalized across heads (last axis).
    :param sentence_labels: 2D tensor, shape [B, num_labels_tok]
    :param num_sent_labels: number of unique sentence labels.
    :param num_tok_labels: number of unique token labels.
    :param minimum_gap_distance: the minimum distance gap imposed between
    scores corresponding tot he default or non-default gold sentence label.
    :param approach: method to extract token representation values.
    :param type_distance: type of gap distance loss that you want.
    :return: a number representing the sum over gap-distance losses.
    """
    if num_sent_labels == num_tok_labels or num_sent_labels == 2:
        # Compute the token representations based on the approach selected.
        token_representative_values = get_token_representative_values(
            token_probabilities, approach)  # [B, num_heads]

        one_hot_sentence_labels = get_one_hot_of_token_labels_length(
            sentence_labels, num_sent_labels, num_tok_labels)
        valid_tokens = tf.multiply(
            tf.cast(one_hot_sentence_labels, tf.float32),
            token_representative_values)  # [B, num_tok_labels]

        tokens_default_head_correct = tf.squeeze(tf.gather(
            valid_tokens, indices=[0], axis=-1), axis=-1)  # [B]
        tokens_default_head_incorrect = tf.squeeze(tf.gather(
            token_representative_values, indices=[0], axis=-1), axis=-1)  # [B]

        tokens_non_default_head_correct = tf.squeeze(
            tf.reduce_max(tf.gather(
                valid_tokens,
                indices=[[i] for i in range(1, num_tok_labels)],
                axis=-1), axis=1), axis=-1)
        tokens_non_default_head_incorrect = tf.squeeze(
            tf.reduce_max(tf.gather(
                token_representative_values,
                indices=[[i] for i in range(1, num_tok_labels)],
                axis=-1), axis=1), axis=-1)

        heads_correct = tf.stack(
            [tokens_default_head_correct, tokens_non_default_head_correct],
            axis=-1)  # [B, 2]
        heads_incorrect = tf.stack(
            [tokens_default_head_incorrect, tokens_non_default_head_incorrect],
            axis=-1)  # [B, 2]
        y_heads = tf.where(
            tf.equal(tf.cast(tokens_non_default_head_correct, tf.int32), 0),
            one_hot_sentence_labels,
            tf.ones_like(one_hot_sentence_labels) - one_hot_sentence_labels)

        """
        heads_correct = tf.where(
            tf.equal(tf.cast(tokens_non_default_head, tf.int32), 0),
            tokens_default_head,
            tokens_non_default_head)

        heads_incorrect = tf.where(
            tf.equal(tf.cast(tokens_default_head, tf.int32), 0),
            tokens_default_head,
            tokens_non_default_head)
        """

        if type_distance == "distance_only":
            # loss = max(0.0, threshold - |correct - incorrect|).
            gap_loss = tf.math.maximum(
                0.0,
                tf.math.subtract(
                    minimum_gap_distance,
                    tf.math.abs(tf.subtract(
                        tokens_default_head_incorrect,
                        tokens_non_default_head_incorrect))))
        elif type_distance == "contrastive":
            squared_euclidean_distance = tf.reduce_sum(
                tf.square(heads_correct - heads_incorrect))
            # loss = y * dist + (1 - y) * max(0.0, threshold - d).
            gap_loss = tf.add(
                tf.multiply(tf.ones_like(y_heads) - y_heads,
                            squared_euclidean_distance),
                tf.multiply(y_heads,
                            tf.maximum(0.0,
                                       minimum_gap_distance - squared_euclidean_distance)))
        else:
            # loss =
            # [exp(max(0.0, threshold - |correct - incorrect|))
            #   * (1.0 + max(correct, incorrect) - x_correct)
            #   * (1.0 + incorrect - min(correct, incorrect))] - 1.0
            gap_loss = tf.subtract(
                tf.math.exp(tf.math.maximum(
                    0.0, minimum_gap_distance - tf.math.abs(heads_correct - heads_incorrect)))
                * tf.add(1.0, tf.math.maximum(heads_correct, heads_incorrect) - heads_correct)
                * tf.add(1.0, heads_incorrect - tf.math.minimum(heads_correct, heads_incorrect)),
                1.0)
    else:
        raise ValueError(
            "You have different number of token labels (%d) and "
            "sentence labels (%d, which is non-binary). "
            "We don't support attention loss for such a case!"
            % (num_tok_labels, num_sent_labels))
    return gap_loss

