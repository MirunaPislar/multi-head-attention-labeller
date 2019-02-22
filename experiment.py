import collections
import gc
import math
import numpy
import os
import random
import sys

try:
    import ConfigParser as configparser
except:
    import configparser

from model import MLTModel
from evaluator import MLTEvaluator, MLTMultiLabelEvaluator


def read_input_files(file_paths, binary_labels, max_sentence_length=-1):
    """
    Reads input files in whitespace-separated format.
    Will split file_paths on comma, reading from multiple files.
    """
    # Last element on the line is the sentence label, thus we need to increase max_sentence_length by 1
    if not binary_labels and max_sentence_length >= 0:
        max_sentence_length += 1
    sentences = []
    line_length = None
    for file_path in file_paths.strip().split(","):
        with open(file_path, "r") as f:
            sentence = []
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    line_parts = line.split()
                    assert(len(line_parts) >= 2), "Line parts less than 2: %s\n" % line
                    assert(len(line_parts) == line_length or line_length is None), "Inconsistent line parts!"
                    line_length = len(line_parts)
                    sentence.append(line_parts)
                elif len(line) == 0 and len(sentence) > 0:
                    if max_sentence_length <= 0 or len(sentence) <= max_sentence_length:
                        if not binary_labels:
                            assert(sentence[-1][0] == "sent_label"), "Sentence label missing!"
                        sentences.append(sentence)
                    sentence = []
            if len(sentence) > 0:
                if max_sentence_length <= 0 or len(sentence) <= max_sentence_length:
                    sentences.append(sentence)
    return sentences


def parse_config(config_section, config_path):
    """
    Reads configuration from the file and returns a dictionary.
    Tries to guess the correct datatype for each of the config values.
    """
    config_parser = configparser.SafeConfigParser(allow_no_value=True)
    config_parser.read(config_path)
    config = collections.OrderedDict()
    for key, value in config_parser.items(config_section):
        if value is None or len(value.strip()) == 0:
            config[key] = None
        elif value.lower() in ["true", "false"]:
            config[key] = config_parser.getboolean(config_section, key)
        elif value.isdigit():
            config[key] = config_parser.getint(config_section, key)
        elif is_float(value):
            config[key] = config_parser.getfloat(config_section, key)
        else:
            config[key] = config_parser.get(config_section, key)
    return config


def is_float(value):
    """
    Check in value is of type float()
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def create_batches_of_sentence_ids(sentences, batch_equal_size, max_batch_size):
    """
    Groups together sentences into batches
    If max_batch_size is positive, this value determines the maximum number of sentences in each batch.
    If max_batch_size has a negative value, the function dynamically creates the batches such that
    each batch contains abs(max_batch_size) words.
    Returns a list of lists with sentences ids.
    """
    batches_of_sentence_ids = []
    if batch_equal_size:
        sentence_ids_by_length = collections.OrderedDict()
        for i in range(len(sentences)):
            length = len(sentences[i])
            if length not in sentence_ids_by_length:
                sentence_ids_by_length[length] = []
            sentence_ids_by_length[length].append(i)

        for sentence_length in sentence_ids_by_length:
            if max_batch_size > 0:
                batch_size = max_batch_size
            else:
                batch_size = int((-1 * max_batch_size) / sentence_length)

            for i in range(0, len(sentence_ids_by_length[sentence_length]), batch_size):
                batches_of_sentence_ids.append(sentence_ids_by_length[sentence_length][i:i + batch_size])
    else:
        current_batch = []
        max_sentence_length = 0
        for i in range(len(sentences)):
            current_batch.append(i)
            if len(sentences[i]) > max_sentence_length:
                max_sentence_length = len(sentences[i])
            if (0 < max_batch_size <= len(current_batch)) \
                    or (max_batch_size <= 0 and len(current_batch) * max_sentence_length >=
                        (-1 * max_batch_size)):
                batches_of_sentence_ids.append(current_batch)
                current_batch = []
                max_sentence_length = 0
        if len(current_batch) > 0:
            batches_of_sentence_ids.append(current_batch)
    return batches_of_sentence_ids


def process_sentences(data, model, is_training, learningrate, config, name):
    """
    Process all the sentences with the labeler, return evaluation metrics.
    """
    if config["binary_labels"]:
        evaluator = MLTEvaluator(config)
    else:
        evaluator = MLTMultiLabelEvaluator(
            config, sent_label2id=model.sent_label2id, token_label2id=model.token_label2id)
    batches_of_sentence_ids = create_batches_of_sentence_ids(
        sentences=data, batch_equal_size=config["batch_equal_size"],
        max_batch_size=config["max_batch_size"])

    if is_training:
        random.shuffle(batches_of_sentence_ids)

    for sentence_ids_in_batch in batches_of_sentence_ids:
        batch = [data[i] for i in sentence_ids_in_batch]
        cost, sentence_scores, token_scores = model.process_batch(batch, is_training, learningrate)
        evaluator.append_data(cost, batch, sentence_scores, token_scores)

        while config["garbage_collection"] and gc.collect() > 0:
            pass

    results = evaluator.get_results(name)
    for key in results:
        print(key + ": " + str(results[key]))

    evaluator.get_scikit_results()
    return results


def run_experiment(config_path):
    config = parse_config("config", config_path)
    temp_model_path = config_path + ".model"
    if "random_seed" in config:
        random.seed(config["random_seed"])
        numpy.random.seed(config["random_seed"])

    for key, val in config.items():
        print(str(key) + ": " + str(val))

    data_train, data_dev, data_test = None, None, None
    if config["path_train"] and len(config["path_train"]) > 0:
        data_train = read_input_files(file_paths=config["path_train"],
                                      max_sentence_length=config["max_train_sent_length"],
                                      binary_labels=config["binary_labels"])
    if config["path_dev"] and len(config["path_dev"]) > 0:
        data_dev = read_input_files(file_paths=config["path_dev"],
                                    binary_labels=config["binary_labels"])
    if config["path_test"] and len(config["path_test"]) > 0:
        data_test = []
        for path_test in config["path_test"].strip().split(":"):
            data_test += read_input_files(file_paths=path_test,
                                          binary_labels=config["binary_labels"])
    data_train = data_train[:100]
    data_dev = data_dev[:50]
    data_test = data_test[:50]

    model = MLTModel(config)
    model.build_vocabs(data_train, data_dev, data_test,
                       embedding_path=config["preload_vectors"])
    model.construct_network()
    model.initialize_session()

    if config["preload_vectors"]:
        model.preload_word_embeddings(config["preload_vectors"])

    print("parameter_count: " + str(model.get_parameter_count()))
    print("parameter_count_without_word_embeddings: " + str(model.get_parameter_count_without_word_embeddings()))

    if data_train:
        model_selector = config["model_selector"].split(":")[0]
        model_selector_type = config["model_selector"].split(":")[1]
        best_selector_value = 0.0
        best_epoch = -1
        learningrate = config["learningrate"]

        for epoch in range(config["epochs"]):
            print("\nEPOCH: " + str(epoch))
            print("current_learningrate: " + str(learningrate))
            random.shuffle(data_train)

            results_train = process_sentences(
                data_train, model,
                is_training=True, learningrate=learningrate,
                config=config, name="train")

            if data_dev:
                results_dev = process_sentences(
                    data_dev, model,
                    is_training=False, learningrate=0.0,
                    config=config, name="dev")

                if math.isnan(results_dev["dev_cost_sum"]) or math.isinf(results_dev["dev_cost_sum"]):
                    raise ValueError("Cost is NaN or Inf. Exiting.")

                if (epoch == 0 or
                        (model_selector_type == "high" and results_dev[model_selector] > best_selector_value) or
                        (model_selector_type == "low" and results_dev[model_selector] < best_selector_value)):
                    best_epoch = epoch
                    best_selector_value = results_dev[model_selector]
                    model.saver.save(model.session, temp_model_path,
                                     latest_filename=os.path.basename(temp_model_path) + ".checkpoint")

                print("best_epoch: " + str(best_epoch))

                if 0 < config["stop_if_no_improvement_for_epochs"] <= (epoch - best_epoch):
                    break

                if (epoch - best_epoch) > 3:
                    learningrate *= config["learningrate_decay"]

            while config["garbage_collection"] and gc.collect() > 0:
                pass

        if data_dev and best_epoch >= 0:
            # Loading the best model so far
            model.saver.restore(model.session, temp_model_path)
            os.remove(temp_model_path+".checkpoint")
            os.remove(temp_model_path+".data-00000-of-00001")
            os.remove(temp_model_path+".index")
            os.remove(temp_model_path+".meta")

    if config["save"] is not None and len(config["save"]) > 0:
        model.save(config["save"])

    if config["path_test"] is not None:
        i = 0
        for path_test in config["path_test"].strip().split(":"):
            data_test = read_input_files(file_paths=path_test,
                                         binary_labels=config["binary_labels"])
            results_test = process_sentences(
                data_test, model, 
                is_training=False, learningrate=0.0, 
                config=config, name="test"+str(i))
            i += 1


if __name__ == "__main__":
    run_experiment(sys.argv[1])
