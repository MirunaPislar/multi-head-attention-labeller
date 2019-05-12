from collections import Counter
from collections import OrderedDict
from my_eval import Evaluator
import numpy as np
import random
import sys
import pandas as pd
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
    labels_tok_dict = {}

    def __init__(self, value, label, enable_supervision):
        self.value = value
        self.label_tok = label
        self.enable_supervision = True
        if enable_supervision == "off":
            self.enable_supervision = False
        self.unique_labels_tok.add(label)
        if label not in self.labels_tok_dict.keys():
            self.labels_tok_dict[label] = 0
        self.labels_tok_dict[label] += 1


class Sentence:
    """
    Representation of a sentence as a list of tokens which are of
    class Token, thus each has a certain value and label.
    Each sentence is assigned a label which can be either inferred
    from its tokens (binary/majority) or specified by the user (so
    the last line is "sent_label" followed by the sentence label).
    """
    unique_labels_sent = set()
    labels_sent_dict = {}

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
            if self.label_sent not in self.labels_sent_dict.keys():
                self.labels_sent_dict[self.label_sent] = 0
            self.labels_sent_dict[self.label_sent] += 1

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

    def run_experiment(self, config_path):
        """
        Run the sequence labeling experiment.
        :type config_path: str
        :rtype: None
        """
        self.config = self.parse_config("config", config_path)
        initialize_writer(self.config["to_write_filename"])

        if "random_seed" in self.config:
            random.seed(self.config["random_seed"])
            np.random.seed(self.config["random_seed"])

        for key, val in self.config.items():
            print(str(key) + " = " + str(val))

        if self.config["path_train"] and len(self.config["path_train"]) > 0:
            data_train = []
            for path_train in self.config["path_train"].strip().split(":"):
                data_train += self.read_input_files(
                    file_paths=path_train,
                    max_sentence_length=self.config["max_train_sent_length"])

        majority_sentence_label = Counter(Sentence.labels_sent_dict).most_common(1)[0][0]
        majority_token_label = Counter(Token.labels_tok_dict).most_common(1)[0][0]

        print("Most common sentence label (as in the train set) = ", majority_sentence_label)
        print("Most common token label (as in the train set) = ", majority_token_label)

        self.label2id_sent = self.create_labels_mapping(Sentence.unique_labels_sent)
        self.label2id_tok = self.create_labels_mapping(Token.unique_labels_tok)
        print("Sentence labels to id: ", self.label2id_sent)
        print("Token labels to id: ", self.label2id_tok)

        df_results = None

        if self.config["path_test"] is not None:
            i = 0
            for path_test in self.config["path_test"].strip().split(":"):
                data_test = self.read_input_files(path_test)
                data_test = self.convert_labels(data_test)

                # Majority baseline.
                majority_pred_sent = [self.label2id_sent[majority_sentence_label]] * len(data_test)
                majority_pred_tok = []
                for sentence in data_test:
                    majority_pred_tok.append(
                        [self.label2id_tok[majority_token_label]] * len(sentence.tokens))

                majority_evaluator = Evaluator(
                    self.label2id_sent, self.label2id_tok, self.config["conll03_eval"])
                majority_evaluator.append_data(
                    0.0, data_test, majority_pred_sent, majority_pred_tok)

                name = "majority_test" + str(i)
                results = majority_evaluator.get_results(
                    name=name, token_labels_available=self.config["token_labels_available"])

                for key in results:
                    print("%s_%s: %s" % (name, key, str(results[key])))
                majority_evaluator.get_results_nice_print(
                    name=name, token_labels_available=self.config["token_labels_available"])

                if df_results is None:
                    df_results = pd.DataFrame(columns=results.keys())
                df_results = df_results.append(results, ignore_index=True)

                # Random baseline.
                random_pred_sent = []
                random_pred_tok = []
                for sentence in data_test:
                    random_pred_sent.append(random.randint(0, len(self.label2id_sent) - 1))
                    random_pred_tok.append(
                        [random.randint(0, len(self.label2id_tok) - 1)
                         for _ in range(len(sentence.tokens))])

                random_evaluator = Evaluator(
                    self.label2id_sent, self.label2id_tok, self.config["conll03_eval"])
                random_evaluator.append_data(
                    0.0, data_test, random_pred_sent, random_pred_tok)

                name = "rand_test" + str(i)
                results = random_evaluator.get_results(
                    name=name, token_labels_available=self.config["token_labels_available"])

                for key in results:
                    print("%s_%s: %s" % (name, key, str(results[key])))
                random_evaluator.get_results_nice_print(
                    name=name, token_labels_available=self.config["token_labels_available"])

                df_results = df_results.append(results, ignore_index=True)
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
    experiment.run_experiment(sys.argv[1])
