from my_second_model import Model
from my_eval import Evaluator
from collections import Counter
from collections import OrderedDict
import gc
import math
import numpy
import os
import random
import sys
import configparser


class Token:
    unique_labels_tok = set()

    def __init__(self, value, label):
        self.value = value
        self.label_tok = label
        self.unique_labels_tok.add(label)


class Sentence:
    unique_labels_sent = set()

    def __init__(self):
        self.tokens = []
        self.label_sent = None

    def add_token(self, value, label, sentence_label_type, default_label):
        """
        Add a token with the specified value and label to the list of tokens.
        If the token value is "sent_label" then instead of adding a token,
        the sentence label is set for which the sentence_label_type and
        the default_label are needed.
        :param value: str
        :param label: str
        :param sentence_label_type: str
        :param default_label: str
        :rtype: None
        """
        if value == "sent_label":
            self.set_label(sentence_label_type, default_label, label)
        else:
            token = Token(value, label)
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
            self.label_sent = Counter(
                [token.label_tok for token in self.tokens]).most_common()[0][0]
        elif label is None and sentence_label_type == "binary":
            non_default_token_labels = sum(
                [0 if token.label_tok == default_label else 1 for token in self.tokens])
            if non_default_token_labels:
                self.label_sent = "1"
            else:
                self.label_sent = "0"  # Experiment.config["default_label"]
        if self.label_sent is not None:
            self.unique_labels_sent.add(self.label_sent)

    def print_sentence(self):
        """
        Print a sentence in this format: "sent_label: tok_i(tok_i label)".
        :rtype: None
        """
        to_print = []
        for token in self.tokens:
            to_print.append("%s(%s)" % (token.value, token.label_tok))
        print("sent %s: %s\n" % (self.label_sent, " ".join(to_print)))


class Experiment:

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
        for file_path in file_paths.strip().split(","):
            with open(file_path, "r") as f:
                sentence = Sentence()
                for line in f:
                    line = line.strip()
                    if len(line) > 0:
                        line_parts = line.split()
                        assert len(line_parts) >= 2, \
                            "Line parts less than 2: %s\n" % line
                        assert len(line_parts) == line_length or line_length is None, \
                            "Inconsistent line parts: expected %d, but got %d." % (
                                len(line_parts), line_length)
                        line_length = len(line_parts)
                        # The first element on the line is the token value, the last is the token label
                        sentence.add_token(
                            value=line_parts[0], label=line_parts[-1],
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
                        sentences.append(sentence)
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
            sentence.label_sent = self.label2id_sent[current_label_sent]
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
        Process the sentences by passing them to the model labeler. 
        Return the evaluation metrics.
        :type sentences: List[Sentence]
        :type model: Model
        :type is_training: bool
        :type learning_rate: float
        :type name: str
        :rtype: List[floats]
        """
        evaluator = Evaluator(self.config, self.label2id_sent, self.label2id_tok)
        batches_of_sentence_ids = self.create_batches_of_sentence_ids(
            sentences,
            self.config["batch_equal_size"],
            self.config["max_batch_size"])

        if is_training:
            random.shuffle(batches_of_sentence_ids)

        for sentence_ids_in_batch in batches_of_sentence_ids:
            batch = [sentences[i] for i in sentence_ids_in_batch]
            cost, sentence_scores, token_scores_list = model.process_batch(
                batch, is_training, learning_rate)
            evaluator.append_data(cost, batch, sentence_scores, token_scores_list)

            while self.config["garbage_collection"] and gc.collect() > 0:
                pass

        results = evaluator.get_results(name)
        for key in results:
            print(key + ": " + str(results[key]))
        evaluator.get_scikit_results()
        return results

    def run_experiment(self, config_path):
        """
        Run the sequence labeling experiment.
        :type config_path: str
        :rtype: None
        """
        self.config = self.parse_config("config", config_path)
        temp_model_path = config_path + ".model"

        if "random_seed" in self.config:
            random.seed(self.config["random_seed"])
            numpy.random.seed(self.config["random_seed"])

        for key, val in self.config.items():
            print(str(key) + ": " + str(val))

        data_train, data_dev, data_test = None, None, None
        if self.config["path_train"] and len(self.config["path_train"]) > 0:
            data_train = self.read_input_files(
                self.config["path_train"], self.config["max_train_sent_length"])
        if self.config["path_dev"] and len(self.config["path_dev"]) > 0:
            data_dev = self.read_input_files(self.config["path_dev"])
        if self.config["path_test"] and len(self.config["path_test"]) > 0:
            data_test = []
            for path_test in self.config["path_test"].strip().split(":"):
                data_test += self.read_input_files(file_paths=path_test)

        self.label2id_sent = self.create_labels_mapping(Sentence.unique_labels_sent)
        self.label2id_tok = self.create_labels_mapping(Token.unique_labels_tok)
        print(self.label2id_sent)
        print(self.label2id_tok)

        data_train = self.convert_labels(data_train)
        data_dev = self.convert_labels(data_dev)
        data_test = self.convert_labels(data_test)

        data_train = data_train[:1000]
        data_dev = data_dev[:500]
        data_test = data_test[:500]

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
        
        if data_train:
            model_selector = self.config["model_selector"].split(":")[0]
            model_selector_type = self.config["model_selector"].split(":")[1]
            best_selector_value = 0.0
            best_epoch = -1
            learning_rate = self.config["learning_rate"]

            for epoch in range(self.config["epochs"]):
                print("*" * 30)
                print("EPOCH: %d" % epoch)
                print("Learning rate: %f" % learning_rate)
                random.shuffle(data_train)

                self.process_sentences(
                    data_train, model, is_training=True,
                    learning_rate=learning_rate, name="train")

                if data_dev:
                    results_dev = self.process_sentences(
                        data_dev, model, is_training=False,
                        learning_rate=0.0, name="dev")

                    if math.isnan(results_dev["dev_cost_sum"]) or math.isinf(results_dev["dev_cost_sum"]):
                        raise ValueError("Cost is NaN or Inf. Exiting.")

                    if (epoch == 0 or
                            (model_selector_type == "high" and results_dev[model_selector] > best_selector_value) or
                            (model_selector_type == "low" and results_dev[model_selector] < best_selector_value)):
                        best_epoch = epoch
                        best_selector_value = results_dev[model_selector]
                        model.saver.save(model.session, temp_model_path,
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
                    self.process_sentences(
                        data_test, model, is_training=False,
                        learning_rate=0.0, name="test" + str(i))
                    i += 1


if __name__ == "__main__":
    experiment = Experiment()
    experiment.run_experiment(sys.argv[1])

