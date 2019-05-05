from math import ceil
import tensorflow as tf


def mask(inputs, queries=None, keys=None, mask_type=None):
    """
    Generate masks and apply them to the inputs.
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
    Generate masks and apply them to the inputs.
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

        # attention_weights_unnormalized = tf.where(
        #     tf.sequence_mask(sentence_lengths),
        #     attention_weights_unnormalized,
        #     tf.zeros_like(attention_weights_unnormalized))

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

