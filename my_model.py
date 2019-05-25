# from marek_model import Model
from my_model import Model
from my_eval import Evaluator
from collections import Counter
from collections import OrderedDict
import gc
import math
import numpy as np
import os
import pandas as pd
import random
import sys
import time
import visualize
import warnings

warnings.filterwarnings("ignore")

if sys.version_info[0] < 3:
    import ConfigParser as configparser
else:
    import configparser


class Token:
    """
    Representation of a single token as a value and a label.
    """
    unique_labels_tok = set()

    def __init__(self, value, label, enable_supervision):
        self.value = value
        self.label_tok = label
        self.enable_supervision = True
        if enable_supervision == "off":
            self.enable_supervision = False
        self.unique_labels_tok.add(label)


class Sentence:
    """
    Representation of a sentence as a list of tokens which are of
    class Token, thus each has a certain value and label.
    Each sentence is assigned a label which can be either inferred
    from its tokens (binary/majority) or specified by the user (so
    the last line is "sent_label" followed by the sentence label).
    """
    unique_labels_sent = set()

    def __init__(self):
        self.tokens = []
        self.label_sent = None

    def add_token(self, value, label, enable_supervision,
                  sentence_label_type, default_label):
        """
        Add a token with the specified value and label to the list of tokens.
        If the token value is "sent_label" then instead of adding a token,
        the sentence label is set for which the sentence_label_type and
        the default_label are needed.
        :param value: str, the actual token value (what is the word, precisely)
        :param label: str, the lable of the current token
        :param enable_supervision: str, whether to allow supervision or not
        :param sentence_label_type: str
        :param default_label: str
        :rtype: None
        """
        if value == "sent_label":
            self.set_label(sentence_label_type, default_label, label)
        else:
            token = Token(value, label, enable_supervision)
            self.tokens.append(token)

    def set_label(self, sentence_label_type, default_label, label=None):
        """
        Set the label of the sentence, according to the sentence_label_type,
        which can be "specified", "majority", or "binary".
        The default label is needed to infer the binary labels.
        :param sentence_label_type: str
        :param default_label: str
        :param label: str
        :rtype: None
        """
        if sentence_label_type == "specified":
            assert label is not None or self.label_sent is not None, "Sentence label missing!"
            if label is not None:
                self.label_sent = label
        elif label is None and sentence_label_type == "majority":
            majority_label = Counter(
                [token.label_tok for token in self.tokens]).most_common()[0][0]
            if majority_label is not None:
                self.label_sent = majority_label
            else:
                raise ValueError("Majority label is None! Sentence tokens: ", self.tokens)
        elif label is None and sentence_label_type == "binary":
            non_default_token_labels = sum(
                [0 if token.label_tok == default_label else 1 for token in self.tokens])
            if non_default_token_labels > 0:
                self.label_sent = "1"
            else:
                self.label_sent = "0"  # default_label
        if self.label_sent is not None:
            self.unique_labels_sent.add(self.label_sent)

    def print_sentence(self):
        """
        Print a sentence in this format: "sent_label: tok_i(label_i, is_supervision_enabled_i)".
        :rtype: int, representing the number of tokens enabled in this sentence
        """
        to_print = []
        num_tokens_enabled = 0
        for token in self.tokens:
            to_print.append("%s (%s, %s)" % (token.value, token.label_tok, token.enable_supervision))
            if token.enable_supervision:
                num_tokens_enabled += 1
        print("sent %s: %s\n" % (self.label_sent, " ".join(to_print)))
        if self.tokens[0].enable_supervision:
            assert num_tokens_enabled == len(self.tokens), \
                "Number of tokens enabled does not equal the number of tokens in the sentence!"
        return num_tokens_enabled


class Experiment:
    """
    Here we start the experiment.
    """

    def __init__(self):
        self.config = None
        self.label2id_sent = None
        self.label2id_tok = None

    def read_input_files(self, file_paths, max_sentence_length=-1):
        """
        Reads input files in whitespace-separated format.
        Will split file_paths on comma, reading from multiple files.
        Expects one token per line: first column = value, last column = label.
        If the sentence label is specified, it needs to have
        first column = "sent_label" and config["sentence_label"] = specified.
        If the sentence label is not specified, it will be inferred from the data
        depending on the value of config["sentence_label"]. Can be majority or binary.
        :type file_paths: str
        :type max_sentence_length: int
        :rtype: list of Sentence objects
        """
        sentences = []
        line_length = None
        sentence = Sentence()

        for file_path in file_paths.strip().split(","):
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    if len(line) > 0:
                        line_parts = line.split()
                        assert len(line_parts) >= 2, \
                            "Line parts less than 2: %s\n" % line
                        assert len(line_parts) == line_length or line_length is None, \
                            "Inconsistent line parts: expected %d, but got %d for line %s." % (
                                len(line_parts), line_length, line)
                        line_length = len(line_parts)

                        # The first element on the line is the token value.
                        # The last is the token label. If there is a penultimate column value
                        # on the line that is equal to either "on" or "off", it indicates
                        # whether supervision on this token is enabled. If there is no such element,
                        # we implicitly assume that supervision is possible and turn it on.
                        sentence.add_token(
                            value=line_parts[0], label=line_parts[-1],
                            enable_supervision=line_parts[-2] if len(line_parts) > 2 else "on",
                            sentence_label_type=self.config["sentence_label"],
                            default_label=self.config["default_label"])
                    elif len(line) == 0 and len(sentence.tokens) > 0:
                        if max_sentence_length <= 0 or len(sentence.tokens) <= max_sentence_length:
                            sentence.set_label(
                                sentence_label_type=self.config["sentence_label"],
                                default_label=self.config["default_label"])
                            sentences.append(sentence)
                        sentence = Sentence()
                if len(sentence.tokens) > 0:
                    if max_sentence_length <= 0 or len(sentence.tokens) <= max_sentence_length:
                        sentence.set_label(
                            sentence_label_type=self.config["sentence_label"],
                            default_label=self.config["default_label"])
                        sentences.append(sentence)
                    sentence = Sentence()
        return sentences

    def create_labels_mapping(self, unique_labels):
        """
        Map a list of U unique labels to an index in [0, U).
        The default label (if it exists and is present) will receive index 0.
        All other labels get the index corresponding to their order.
        :type unique_labels: set
        :rtype: dict
        """
        if self.config["default_label"] and self.config["default_label"] in unique_labels:
            sorted_labels = sorted(list(unique_labels.difference(self.config["default_label"])))
            label2id = {label: index + 1 for index, label in enumerate(sorted_labels)}
            label2id[self.config["default_label"]] = 0
        else:
            sorted_labels = sorted(list(unique_labels))
            label2id = {label: index for index, label in enumerate(sorted_labels)}
        return label2id

    def convert_labels(self, data):
        """
        Convert each sentence and token label to its corresponding index.
        :type data: list[Sentence]
        :rtype: list[Sentence]
        """
        for sentence in data:
            current_label_sent = sentence.label_sent
            try:
                sentence.label_sent = self.label2id_sent[current_label_sent]
            except KeyError:
                print("Key error for ", current_label_sent)
                print("Sentence: ", [token.value for token in sentence.tokens])
                print("Label to id", self.label2id_sent)
            for token in sentence.tokens:
                current_label_tok = token.label_tok
                token.label_tok = self.label2id_tok[current_label_tok]
        return data

    def parse_config(self, config_section, config_path):
        """
        Reads the configuration file, guessing the correct data type for each value.
        :type config_section: str
        :type config_path: str
        :rtype: dict
        """
        config_parser = configparser.ConfigParser(allow_no_value=True)
        config_parser.read(config_path)
        config = OrderedDict()
        for key, value in config_parser.items(config_section):
            if value is None or len(value.strip()) == 0:
                config[key] = None
            elif value.lower() in ["true", "false"]:
                config[key] = config_parser.getboolean(config_section, key)
            elif value.isdigit():
                config[key] = config_parser.getint(config_section, key)
            elif self.is_float(value):
                config[key] = config_parser.getfloat(config_section, key)
            else:
                config[key] = config_parser.get(config_section, key)
        return config

    @staticmethod
    def is_float(value):
        """
        Check if value is of type float.
        :type value: any type
        :rtype: bool
        """
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def create_batches_of_sentence_ids(sentences, batch_equal_size, max_batch_size):
        """
        Create batches of sentence ids. A positive max_batch_size determines
         the maximum number of sentences in each batch. A negative max_batch_size 
         dynamically creates the batches such that each batch contains 
         abs(max_batch_size) words. Returns a list of lists with sentences ids.
        :type sentences: List[Sentence]
        :type batch_equal_size: bool
        :type max_batch_size: int
        :rtype: List[List[int]]
        """
        batches_of_sentence_ids = []
        if batch_equal_size:
            sentence_ids_by_length = OrderedDict()
            for i in range(len(sentences)):
                length = len(sentences[i].tokens)
                if length not in sentence_ids_by_length:
                    sentence_ids_by_length[length] = []
                sentence_ids_by_length[length].append(i)

            for sentence_length in sentence_ids_by_length:
                if max_batch_size > 0:
                    batch_size = max_batch_size
                else:
                    batch_size = int((-1 * max_batch_size) / sentence_length)

                for i in range(0, len(sentence_ids_by_length[sentence_length]), batch_size):
                    batches_of_sentence_ids.append(
                        sentence_ids_by_length[sentence_length][i:i + batch_size])
        else:
            current_batch = []
            max_sentence_length = 0
            for i in range(len(sentences)):
                current_batch.append(i)
                if len(sentences[i].tokens) > max_sentence_length:
                    max_sentence_length = len(sentences[i].tokens)
                if ((0 < max_batch_size <= len(current_batch))
                    or (max_batch_size <= 0
                        and len(current_batch) * max_sentence_length >= (-1 * max_batch_size))):
                    batches_of_sentence_ids.append(current_batch)
                    current_batch = []
                    max_sentence_length = 0
            if len(current_batch) > 0:
                batches_of_sentence_ids.append(current_batch)
        return batches_of_sentence_ids

    def process_sentences(self, sentences, model, is_training, learning_rate, name):
        """
        Obtain sentence and token predictions for the sentences.
        Return the evaluation metrics.
        :type sentences: List[Sentence]
        :type model: Model
        :type is_training: bool
        :type learning_rate: float
        :type name: str
        :rtype: List[floats]
        """
        evaluator = Evaluator(self.label2id_sent, self.label2id_tok,
                              self.config["conll03_eval"])

        batches_of_sentence_ids = self.create_batches_of_sentence_ids(
            sentences, self.config["batch_equal_size"], self.config["max_batch_size"])

        if is_training:
            random.shuffle(batches_of_sentence_ids)

        all_batches, all_sentence_probs, all_token_probs = [], [], []

        for batch_of_sentence_ids in batches_of_sentence_ids:
            batch = [sentences[i] for i in batch_of_sentence_ids]

            cost, sentence_pred, sentence_probs, token_pred, token_probs = \
                model.process_batch(batch, is_training, learning_rate)
            evaluator.append_data(cost, batch, sentence_pred, token_pred)

            if "test" in name and self.config["plot_predictions_html"]:
                all_batches.append(batch)
                all_sentence_probs.append(sentence_probs)
                all_token_probs.append(token_probs)

            # Plot the token scores for each sentence in the batch.
            if "test" in name and self.config["plot_token_scores"]:
                for sentence, token_proba_per_sentence in zip(batch, token_probs):
                    visualize.plot_token_scores(
                        token_probs=token_proba_per_sentence,
                        sentence=sentence,
                        head_labels=evaluator.id2label_tok,
                        plot_name="plots/token_vis/token_heads_vis")

            while self.config["garbage_collection"] and gc.collect() > 0:
                pass

        results = evaluator.get_results(
            name=name, token_labels_available=self.config["token_labels_available"])

        for key in results:
            print("%s_%s: %s" % (name, key, str(results[key])))
        evaluator.get_results_nice_print(
            name=name, token_labels_available=self.config["token_labels_available"])

        # Create html visualizations based on the test set predictions.
        if "test" in name and self.config["plot_predictions_html"]:
            save_name = (self.config["to_write_filename"].split("/")[-1]).split(".")[0]
            visualize.plot_predictions(
                all_sentences=all_batches,
                all_sentence_probs=all_sentence_probs,
                all_token_probs=all_token_probs,
                html_name="plots/html_vis/%s" % save_name,
                sent_binary=len(self.label2id_sent) == 2)

        return results

    def run_experiment(self, config_path):
        """
        Run the sequence labeling experiment.
        :type config_path: str
        :rtype: None
        """
        self.config = self.parse_config("config", config_path)
        initialize_writer(self.config["to_write_filename"])
        i_rand = random.randint(1, 10000)
        print("i_rand = ", i_rand)
        temp_model_path = "models/temp_model_%d" % (
            int(time.time()) + i_rand) + ".model"

        if "random_seed" in self.config:
            random.seed(self.config["random_seed"])
            np.random.seed(self.config["random_seed"])

        for key, val in self.config.items():
            print(str(key) + " = " + str(val))

        data_train, data_dev, data_test = None, None, None

        if self.config["path_train"] and len(self.config["path_train"]) > 0:
            data_train = []
            for path_train in self.config["path_train"].strip().split(":"):
                data_train += self.read_input_files(
                    file_paths=path_train,
                    max_sentence_length=self.config["max_train_sent_length"])

        if self.config["path_dev"] and len(self.config["path_dev"]) > 0:
            data_dev = []
            for path_dev in self.config["path_dev"].strip().split(":"):
                data_dev += self.read_input_files(file_paths=path_dev)

        if self.config["path_test"] and len(self.config["path_test"]) > 0:
            data_test = []
            for path_test in self.config["path_test"].strip().split(":"):
                data_test += self.read_input_files(file_paths=path_test)

        self.label2id_sent = self.create_labels_mapping(Sentence.unique_labels_sent)
        self.label2id_tok = self.create_labels_mapping(Token.unique_labels_tok)
        print("Sentence labels to id: ", self.label2id_sent)
        print("Token labels to id: ", self.label2id_tok)

        data_train = self.convert_labels(data_train) if data_train else None
        data_dev = self.convert_labels(data_dev) if data_dev else None
        data_test = self.convert_labels(data_test) if data_test else None

        # data_train = data_train[:500]
        # data_dev = data_dev[:200]
        # data_test = data_test[:100]

        model = Model(self.config, self.label2id_sent, self.label2id_tok)
        model.build_vocabs(data_train, data_dev, data_test,
                           embedding_path=self.config["preload_vectors"])
        model.construct_network()
        model.initialize_session()
        
        if self.config["preload_vectors"]:
            model.preload_word_embeddings(self.config["preload_vectors"])

        print("Parameter count: %d."
              % model.get_parameter_count())
        print("Parameter count without word embeddings: %d."
              % model.get_parameter_count_without_word_embeddings())
        
        if data_train is None:
            raise ValueError("No training set provided!")

        model_selector_splits = self.config["model_selector"].split(":")
        if type(self.config["model_selector_ratio"]) == str:
            model_selector_ratios_splits = [
                float(val) for val in self.config["model_selector_ratio"].split(":")]
        else:
            model_selector_ratios_splits = [self.config["model_selector_ratio"]]
        model_selector_type = model_selector_splits[-1]
        model_selector_values = model_selector_splits[:-1]
        assert (len(model_selector_values) == len(model_selector_ratios_splits)
                or len(model_selector_ratios_splits) == 1), \
            "Model selector values and ratios don't match!"

        # Each model_selector_value contributes in proportion to its
        # corresponding (normalized) weight value. If just one ratio is specified,
        # all model_selector_values receive equal weight.
        if len(model_selector_ratios_splits) == 1:
            normalized_ratio = model_selector_ratios_splits[0] / sum(
                model_selector_ratios_splits * len(model_selector_values))
            model_selector_to_ratio = {value: normalized_ratio for value in model_selector_values}
        else:
            sum_ratios = sum(model_selector_ratios_splits)
            normalized_ratios = [ratio / sum_ratios for ratio in model_selector_ratios_splits]
            model_selector_to_ratio = {value: ratio for value, ratio in
                                       zip(model_selector_values, normalized_ratios)}

        best_selector_value = 0.0
        if model_selector_type == "low":
            best_selector_value = float("inf")
        best_epoch = -1
        learning_rate = self.config["learning_rate"]

        df_results = None

        for epoch in range(self.config["epochs"]):
            print("EPOCH: %d" % epoch)
            print("Learning rate: %f" % learning_rate)
            random.shuffle(data_train)

            results_train = self.process_sentences(
                data_train, model, is_training=True,
                learning_rate=learning_rate, name="train_epoch%d" % epoch)

            if df_results is None:
                df_results = pd.DataFrame(columns=results_train.keys())
            df_results = df_results.append(results_train, ignore_index=True)

            if data_dev:
                results_dev = self.process_sentences(
                    data_dev, model, is_training=False,
                    learning_rate=0.0, name="dev_epoch%d" % epoch)

                df_results = df_results.append(results_dev, ignore_index=True)

                if math.isnan(results_dev["cost_sum"]) or math.isinf(results_dev["cost_sum"]):
                    raise ValueError("Cost is NaN or Inf!")

                results_dev_for_model_selector = sum([
                    results_dev[model_selector] * ratio
                    for model_selector, ratio in model_selector_to_ratio.items()])

                if (epoch == 0 
                    or (model_selector_type == "high" 
                        and results_dev_for_model_selector > best_selector_value) 
                    or (model_selector_type == "low"
                        and results_dev_for_model_selector < best_selector_value)):
                    best_epoch = epoch
                    best_selector_value = results_dev_for_model_selector
                    model.saver.save(sess=model.session, save_path=temp_model_path,
                                     latest_filename=os.path.basename(temp_model_path) + ".checkpoint")

                print("Best epoch: %d" % best_epoch)
                print("*" * 50 + "\n")

                if 0 < self.config["stop_if_no_improvement_for_epochs"] <= epoch - best_epoch:
                    break

                if epoch - best_epoch > 3:
                    learning_rate *= self.config["learning_rate_decay"]

            while self.config["garbage_collection"] and gc.collect() > 0:
                pass

        if data_dev and best_epoch >= 0:
            model.saver.restore(model.session, temp_model_path)
            os.remove(temp_model_path + ".checkpoint")
            os.remove(temp_model_path + ".data-00000-of-00001")
            os.remove(temp_model_path + ".index")
            os.remove(temp_model_path + ".meta")

        if self.config["save"] is not None and len(self.config["save"]) > 0:
            model.save(self.config["save"])

        if self.config["path_test"] is not None:
            i = 0
            for path_test in self.config["path_test"].strip().split(":"):
                data_test = self.read_input_files(path_test)
                data_test = self.convert_labels(data_test)
                # data_test = data_test[:500]
                results_test = self.process_sentences(
                    data_test, model, is_training=False,
                    learning_rate=0.0, name="test" + str(i))
                df_results = df_results.append(results_test, ignore_index=True)
                i += 1

        # Save data frame with all the training and testing results
        df_results.to_csv("".join(self.config["to_write_filename"].split(".")[:-1]) + "_df_results.txt",
                          index=False, sep="\t", encoding="utf-8")


class Writer:
    """
    This class allows me to print to file and to std output at the same time.
    """
    def __init__(self, *writers):
        self.writers = writers

    def write(self, text):
        for w in self.writers:
            w.write(text)

    def flush(self):
        pass


def initialize_writer(to_write_filename):
    """
    Method that initializes my writer class.
    :param to_write_filename: name of the file where the output will be written.
    :return: None.
    """
    file_out = open(to_write_filename, "wt")
    sys.stdout = Writer(sys.stdout, file_out)


if __name__ == "__main__":
    experiment = Experiment()
    load_pretrained = False

    if not load_pretrained:
        experiment.run_experiment(sys.argv[1])
    else:
        experiment.config = experiment.parse_config("config", sys.argv[1])
        filename = experiment.config["save"]
        print("Loaded model %s" % filename)
        loaded_model = Model.load(filename)

        experiment.label2id_sent = loaded_model.label2id_sent
        experiment.label2id_tok = loaded_model.label2id_tok
        print("Sentence labels to id: ", experiment.label2id_sent)
        print("Token labels to id: ", experiment.label2id_tok)

        if experiment.config["path_test"] is not None:
            d = 0
            for path_data_test in experiment.config["path_test"].strip().split(":"):
                data_test_loaded = experiment.read_input_files(path_data_test)
                data_test_loaded = experiment.convert_labels(data_test_loaded)
                # data_test_loaded = data_test_loaded[:250]
                experiment.process_sentences(
                    data_test_loaded, loaded_model, is_training=False,
                    learning_rate=0.0, name="test" + str(d))
                d += 1
