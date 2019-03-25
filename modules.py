from math import ceil
import tensorflow as tf


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
                          attention_size, hidden_units,
                          num_sentence_labels, num_heads,
                          is_training, shrink_input, dropout,
                          use_residual_connection,
                          token_scoring_method):
    """
    Compute the multi-head transformer architecture.
    :param inputs: 3D floats of shape [B x M x E]
    :param initializer: type of initializer (best if glorot or xavier)
    :param attention_size: number of units to use for the attention evidence
    :param hidden_units: number of units to use for the processed vector
    :param num_sentence_labels: number of unique sentence labels
    :param num_heads: number of unique token labels
    :param is_training: if set to True, the current phase is a training one (rather than testing)
    :param shrink_input: if the input should be shrinked or not to attention_size
    :param dropout: the keep_probs value for the dropout
    :param use_residual_connection: if set to True, a residual connection is added to the inputs
    :return sentence_scores: 2D floats of shape [B x num_sentence_labels]
    :return sentence_predictions: predicted labels for each sentence in the batch; ints of shape [B]
    :return token_scores: 3D floats of shape [B x M x num_heads]
    :return token_predictions: predicted labels for each token in each sentence; ints of shape [B x M]
    """
    with tf.variable_scope("transformer_multi_head_attention"):
        if shrink_input:
            num_units = ceil(attention_size / num_heads) * num_heads
        else:
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
            x=attention_weights, rate=1 - dropout_attention,
            name="dropout_transducer_attention_weights")    # [(heads * B) x M x M]

        product = tf.matmul(attention_weights, values)  # [(heads * B) x M x (num_units / heads)]
        product = tf.concat(tf.split(product, num_heads, axis=0), axis=2)  # [B x M x num_units]
        if use_residual_connection:
            product += adapted_inputs  # add a residual connection
            product = normalize(product)  # [B x M x num_units]
        processed_tensor = tf.reduce_sum(product, axis=1)  # [B x num_units]

        if hidden_units > 0:
            processed_tensor = tf.layers.dense(
                inputs=processed_tensor, units=hidden_units,
                activation=tf.tanh, kernel_initializer=initializer)  # [B x hidden_units]

        sentence_scores = tf.layers.dense(
            inputs=processed_tensor, units=num_sentence_labels,
            kernel_initializer=initializer,
            name="output_sent_specified_scores_ff")  # [B x num_unique_sent_labels]
        sentence_proba = tf.nn.softmax(sentence_scores)
        sentence_predictions = tf.argmax(sentence_proba, axis=1)  # [B]

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


def normalize(inputs, epsilon=1e-8):
    with tf.variable_scope("norm"):
        params_shape = inputs.get_shape()[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
        outputs = tf.add(tf.multiply(gamma, normalized), beta)
    return outputs

