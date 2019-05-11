from collections import OrderedDict
from sklearn.metrics import classification_report
import conlleval
import numpy as np
import time


class Evaluator:

    def __init__(self, label2id_sent, label2id_tok, conll03_eval):
        self.id2label_sent = {v: k for k, v in label2id_sent.items()}
        self.id2label_tok = {v: k for k, v in label2id_tok.items()}

        self.conll03_eval = conll03_eval
        self.conll_format = []

        self.true_sent = []
        self.pred_sent = []
        self.true_tok = []
        self.pred_tok = []

        self.cost_sum = 0.0
        self.count_sent = 0.0
        self.correct_binary_sent = 0.0
        self.count_tok = 0.0
        self.correct_binary_tok = 0.0

        self.sentence_predicted = {k: 0.0 for k in self.id2label_sent.keys()}
        self.sentence_correct = {k: 0.0 for k in self.id2label_sent.keys()}
        self.sentence_total = {k: 0.0 for k in self.id2label_sent.keys()}

        self.token_predicted = {k: 0.0 for k in self.id2label_tok.keys()}
        self.token_correct = {k: 0.0 for k in self.id2label_tok.keys()}
        self.token_total = {k: 0.0 for k in self.id2label_tok.keys()}

        self.start_time = time.time()

    def append_token_data_for_sentence(self, tokens, true_labels_tok, pred_labels_tok):
        """
        Get statistical results from the tokens in a certain sentence.
        """
        self.count_tok += len(true_labels_tok)

        # For each token, calculate the same metrics as for the sentence scores
        for token, true_label, pred_label in zip(tokens, true_labels_tok, pred_labels_tok):
            self.true_tok.append(true_label)
            self.pred_tok.append(pred_label)

            # Calculate accuracy.
            if true_label == pred_label:
                self.correct_binary_tok += 1.0

            # Calculate TP + FP.
            self.token_predicted[pred_label] += 1.0

            # Calculate TP + FN.
            self.token_total[true_label] += 1.0

            # Calculate TP.
            if true_label == pred_label:
                self.token_correct[true_label] += 1.0

            if self.conll03_eval is True:
                gold_token_label = self.id2label_tok[true_label]
                gold_token_label = "B-" + gold_token_label if true_label != 0 else gold_token_label
                pred_token_label = self.id2label_tok[pred_label]
                pred_token_label = "B-" + pred_token_label if true_label != 0 else pred_token_label
                self.conll_format.append(
                    token + "\t" + gold_token_label + "\t" + pred_token_label)
        if self.conll03_eval is True:
            self.conll_format.append("")

    def append_data(self, cost, batch, sentence_predictions, token_predictions):
        """
        Get statistical results from the sentence and token scores in a certain batch.
        """
        self.cost_sum += cost
        self.count_sent += len(batch)

        for i, sentence in enumerate(batch):
            true_labels_tok = [token.label_tok for token in sentence.tokens]
            true_labels_sent = sentence.label_sent
            self.true_sent.append(true_labels_sent)
            self.pred_sent.append(sentence_predictions[i])

            # Calculate accuracy.
            if true_labels_sent == sentence_predictions[i]:
                self.correct_binary_sent += 1.0

            # Calculate TP + FP.
            self.sentence_predicted[sentence_predictions[i]] += 1.0

            # Calculate TP + FN.
            self.sentence_total[true_labels_sent] += 1.0

            # Calculate TP.
            if true_labels_sent == sentence_predictions[i]:
                self.sentence_correct[true_labels_sent] += 1.0

            # Get the scores for the tokens in this sentence
            self.append_token_data_for_sentence(
                [token.value for token in sentence.tokens],
                true_labels_tok, list(token_predictions[i])[:len(true_labels_tok)])

    @staticmethod
    def calculate_metrics(correct, predicted, total):
        """
        Calculate the basic metrics.
        :param correct: the number of examples predicted as correct that are actually correct
        :param predicted: the number of examples predicted as correct.
        :param total: the number of examples that are correct by the gold standard.
        :return: the precision, recall, F1 and F05 scores
        """
        p = correct / predicted if predicted else 0.0
        r = correct / total if total else 0.0
        f = 2.0 * p * r / (p + r) if p + r else 0.0
        f05 = (1 + 0.5 * 0.5) * p * r / (0.5 * 0.5 * p + r) if 0.5 * 0.5 * p + r else 0.0
        return p, r, f, f05

    def get_results(self, name, token_labels_available=True):
        """
        Obtain the statistical results for a certain dataset
        both at the sentence and at the token level.
        :param name: train, dev or test
        :param token_labels_available: whether there are token annotations
        :return: an ordered dictionary containing the collection of results
        """
        results = OrderedDict()

        results["name"] = name
        results["cost_sum"] = self.cost_sum
        results["cost_avg"] = (self.cost_sum / float(self.count_sent)
                               if self.count_sent else 0.0)

        results["count_sent"] = self.count_sent
        results["total_correct_sent"] = self.correct_binary_sent
        results["accuracy_sent"] = (self.correct_binary_sent / float(self.count_sent)
                                    if self.count_sent else 0.0)

        # Calculate the micro and macro averages for the sentence predictions
        f_macro_sent, p_macro_sent, r_macro_sent, f05_macro_sent = 0.0, 0.0, 0.0, 0.0
        f_non_default_macro_sent, p_non_default_macro_sent, \
            r_non_default_macro_sent, f05_non_default_macro_sent = 0.0, 0.0, 0.0, 0.0

        for key in self.id2label_sent.keys():
            p, r, f, f05 = self.calculate_metrics(
                self.sentence_correct[key], self.sentence_predicted[key], self.sentence_total[key])
            label = "label=%s" % self.id2label_sent[key]
            results[label + "_predicted_sent"] = self.sentence_predicted[key]
            results[label + "_correct_sent"] = self.sentence_correct[key]
            results[label + "_total_sent"] = self.sentence_total[key]
            results[label + "_precision_sent"] = p
            results[label + "_recall_sent"] = r
            results[label + "_f-score_sent"] = f
            results[label + "_f05-score_sent"] = f05
            p_macro_sent += p
            r_macro_sent += r
            f_macro_sent += f
            f05_macro_sent += f05
            if key != 0:
                p_non_default_macro_sent += p
                r_non_default_macro_sent += r
                f_non_default_macro_sent += f
                f05_non_default_macro_sent += f05

        p_macro_sent /= len(self.id2label_sent.keys())
        r_macro_sent /= len(self.id2label_sent.keys())
        f_macro_sent /= len(self.id2label_sent.keys())
        f05_macro_sent /= len(self.id2label_sent.keys())

        p_non_default_macro_sent /= (len(self.id2label_sent.keys()) - 1)
        r_non_default_macro_sent /= (len(self.id2label_sent.keys()) - 1)
        f_non_default_macro_sent /= (len(self.id2label_sent.keys()) - 1)
        f05_non_default_macro_sent /= (len(self.id2label_sent.keys()) - 1)

        p_micro_sent, r_micro_sent, f_micro_sent, f05_micro_sent = self.calculate_metrics(
            sum(self.sentence_correct.values()),
            sum(self.sentence_predicted.values()),
            sum(self.sentence_total.values()))

        p_non_default_micro_sent, r_non_default_micro_sent, \
            f_non_default_micro_sent, f05_non_default_micro_sent = self.calculate_metrics(
                sum([value for key, value in self.sentence_correct.items() if key != 0]),
                sum([value for key, value in self.sentence_predicted.items() if key != 0]),
                sum([value for key, value in self.sentence_total.items() if key != 0]))

        results["precision_macro_sent"] = p_macro_sent
        results["recall_macro_sent"] = r_macro_sent
        results["f-score_macro_sent"] = f_macro_sent
        results["f05-score_macro_sent"] = f05_macro_sent

        results["precision_micro_sent"] = p_micro_sent
        results["recall_micro_sent"] = r_micro_sent
        results["f-score_micro_sent"] = f_micro_sent
        results["f05-score_micro_sent"] = f05_micro_sent

        results["precision_non_default_macro_sent"] = p_non_default_macro_sent
        results["recall_non_default_macro_sent"] = r_non_default_macro_sent
        results["f-score_non_default_macro_sent"] = f_non_default_macro_sent
        results["f05-score_non_default_macro_sent"] = f05_non_default_macro_sent

        results["precision_non_default_micro_sent"] = p_non_default_micro_sent
        results["recall_non_default_micro_sent"] = r_non_default_micro_sent
        results["f-score_non_default_micro_sent"] = f_non_default_micro_sent
        results["f05-score_non_default_micro_sent"] = f05_non_default_micro_sent

        if token_labels_available or "test" in name:
            results["count_tok"] = self.count_tok
            results["total_correct_tok"] = self.correct_binary_tok
            results["accuracy_tok"] = (self.correct_binary_tok / float(self.count_tok)
                                       if self.count_tok else 0.0)

            # Calculate the micro and macro averages for the token predictions
            f_tok_macro, p_tok_macro, r_tok_macro, f05_tok_macro = 0.0, 0.0, 0.0, 0.0
            f_non_default_macro_tok, p_non_default_macro_tok, \
                r_non_default_macro_tok, f05_non_default_macro_tok = 0.0, 0.0, 0.0, 0.0

            for key in self.id2label_tok.keys():
                p, r, f, f05 = self.calculate_metrics(
                    self.token_correct[key], self.token_predicted[key], self.token_total[key])
                label = "label=%s" % self.id2label_tok[key]
                results[label + "_predicted_tok"] = self.token_predicted[key]
                results[label + "_correct_tok"] = self.token_correct[key]
                results[label + "_total_tok"] = self.token_total[key]
                results[label + "_precision_tok"] = p
                results[label + "_recall_tok"] = r
                results[label + "_f-score_tok"] = f
                results[label + "_tok_f05"] = f05
                p_tok_macro += p
                r_tok_macro += r
                f_tok_macro += f
                f05_tok_macro += f05
                if key != 0:
                    p_non_default_macro_tok += p
                    r_non_default_macro_tok += r
                    f_non_default_macro_tok += f
                    f05_non_default_macro_tok += f05

            p_tok_macro /= len(self.id2label_tok.keys())
            r_tok_macro /= len(self.id2label_tok.keys())
            f_tok_macro /= len(self.id2label_tok.keys())
            f05_tok_macro /= len(self.id2label_tok.keys())

            p_non_default_macro_tok /= (len(self.id2label_tok.keys()) - 1)
            r_non_default_macro_tok /= (len(self.id2label_tok.keys()) - 1)
            f_non_default_macro_tok /= (len(self.id2label_tok.keys()) - 1)
            f05_non_default_macro_tok /= (len(self.id2label_tok.keys()) - 1)

            p_tok_micro, r_tok_micro, f_tok_micro, f05_tok_micro = self.calculate_metrics(
                sum(self.token_correct.values()),
                sum(self.token_predicted.values()),
                sum(self.token_total.values()))

            p_non_default_micro_tok, r_non_default_micro_tok, \
                f_non_default_micro_tok, f05_non_default_micro_tok = self.calculate_metrics(
                    sum([value for key, value in self.token_correct.items() if key != 0]),
                    sum([value for key, value in self.token_predicted.items() if key != 0]),
                    sum([value for key, value in self.token_total.items() if key != 0]))

            results["precision_macro_tok"] = p_tok_macro
            results["recall_macro_tok"] = r_tok_macro
            results["f-score_macro_tok"] = f_tok_macro
            results["f05-score_macro_tok"] = f05_tok_macro

            results["precision_micro_tok"] = p_tok_micro
            results["recall_micro_tok"] = r_tok_micro
            results["f-score_micro_tok"] = f_tok_micro
            results["f05-score_micro_tok"] = f05_tok_micro

            results["precision_non_default_macro_tok"] = p_non_default_macro_tok
            results["recall_non_default_macro_tok"] = r_non_default_macro_tok
            results["f-score_non_default_macro_tok"] = f_non_default_macro_tok
            results["f05-score_non_default_macro_tok"] = f05_non_default_macro_tok

            results["precision_non_default_micro_tok"] = p_non_default_micro_tok
            results["recall_non_default_micro_tok"] = r_non_default_micro_tok
            results["f-score_non_default_micro_tok"] = f_non_default_micro_tok
            results["f05-score_non_default_micro_tok"] = f05_non_default_micro_tok

            if self.id2label_tok is not None and self.conll03_eval is True:
                conll_counts = conlleval.evaluate(self.conll_format)
                conll_metrics_overall, conll_metrics_by_type = conlleval.metrics(conll_counts)
                results["conll_accuracy"] = (float(conll_counts.correct_tags)
                                             / float(conll_counts.token_counter))
                results["conll_p"] = conll_metrics_overall.prec
                results["conll_r"] = conll_metrics_overall.rec
                results["conll_f"] = conll_metrics_overall.fscore

        results["time"] = float(time.time()) - float(self.start_time)
        return results

    def get_results_nice_print(self, name, token_labels_available=True):
        """
        This method is just a wrapper around the statistical results already computed,
        just to print them bolder and nicer. Can also used to check the basic metrics.
        :return: nothing, just print a classification report for the tokens and sentences.
        """
        if self.true_sent and self.pred_sent:
            print("*" * 50)
            print("Sentence predictions: ")
            print(classification_report(
                self.true_sent, self.pred_sent, digits=4, labels=np.array(range(len(self.id2label_sent))),
                target_names=[self.id2label_sent[i] for i in range(len(self.id2label_sent))]))

        if token_labels_available or "test" in name:
            if self.true_tok and self.pred_tok:
                print("*" * 50)
                print("Token predictions: ")
                print(classification_report(
                    self.true_tok, self.pred_tok, digits=4, labels=np.array(range(len(self.id2label_tok))),
                    target_names=[self.id2label_tok[i] for i in range(len(self.id2label_tok))]))

