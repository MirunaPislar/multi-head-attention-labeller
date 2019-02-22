from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import collections
import numpy
import time


class MLTEvaluator(object):
    def __init__(self, config):
        self.config = config

        self.sent_true = []
        self.sent_pred = []
        self.tok_true = []
        self.tok_pred = []

        self.cost_sum = 0.0
        self.sentence_count = 0.0
        self.sentence_correct_binary = 0.0
        self.sentence_predicted = 0.0
        self.sentence_correct = 0.0
        self.sentence_total = 0.0

        self.token_ap_sum = []
        self.token_predicted = []
        self.token_correct = []
        self.token_total = []

        self.start_time = time.time()

    def calculate_ap(self, true_labels, predicted_scores):
        assert(len(true_labels) == len(predicted_scores))

        if self.config["debug_mode"]:
            with open("my_output.txt", "a") as f:
                f.write("\nTrue labels: \n" + str(" ".join([str(t) for t in true_labels])))
                f.write("\nPredicted labels: \n" + str(" ".join([str(t) for t in predicted_scores])))

        # This orders the predicted scores and returns the indices, in reverse order (biggest to smallest)
        # Here: but why do we do this?
        indices = numpy.argsort(numpy.array(predicted_scores))[::-1]

        summed, correct, total = 0.0, 0.0, 0.0
        for index in indices:
            total += 1.0
            if true_labels[index] >= 0.5:
                correct += 1.0
                summed += correct / total
        return (summed / correct) if correct > 0.0 else 0.0

    def append_token_data_for_sentence(self, index, true_labels, token_scores):
        if len(self.token_ap_sum) <= index:
            self.token_ap_sum.append(0.0)
            self.token_predicted.append(0.0)
            self.token_correct.append(0.0)
            self.token_total.append(0.0)

        ap = self.calculate_ap(true_labels, token_scores[:len(true_labels)])
        self.token_ap_sum[index] += ap

        # For each token in the sentence, calculate the same metrics as for the sentence scores
        for i in range(len(true_labels)):

            self.tok_true.append(true_labels[i])
            self.tok_pred.append(int(token_scores[i]))

            # Total number of tokens with predicted label 1 (non-default label) = TP + FP
            if token_scores[i] == 1:
                self.token_predicted[index] += 1.0
                # Total number of tokens with true label 1 (non-default label) = TP + FN
            if true_labels[i] == 1:
                self.token_total[index] += 1.0
                # Total number of tokens with both predicted and true labels 1 (non-default label) = TP
            if true_labels[i] == 1 and token_scores[i] == 1:
                self.token_correct[index] += 1.0
            """
            if token_scores[i] >= 0.5:
                self.token_predicted[index] += 1.0
            # Total number of tokens with true label 1 (non-default label) = TP + FN
            if true_labels[i] >= 0.5:
                self.token_total[index] += 1.0
            # Total number of tokens with both predicted and true labels 1 (non-default label) = TP
            if true_labels[i] >= 0.5 and token_scores[i] >= 0.5:
                self.token_correct[index] += 1.0
            """

    def append_data(self, cost, batch, sentence_scores, token_scores):
        if self.config["binary_labels"]:
            assert (len(self.token_ap_sum) == 0 or len(self.token_ap_sum) == len(token_scores))
            self.cost_sum += cost

            for i in range(len(batch)):
                self.sentence_count += 1.0
                # HERE: this is the same as the labels in model, why not re-use so no mistakes made
                true_labels = [0.0 if batch[i][j][-1] == self.config["default_label"]
                               else 1.0 for j in range(len(batch[i]))]
                count_interesting_labels = numpy.array(true_labels).sum()

                if count_interesting_labels == 0:
                    true_sent_lab = 0
                else:
                    true_sent_lab = 1

                self.sent_true.append(true_sent_lab)
                self.sent_pred.append(sentence_scores[i])

                # Total number of sentence labels for which the correct label was assigned (accuracy)
                if (count_interesting_labels == 0.0 and sentence_scores[i] < 0.5) \
                        or (count_interesting_labels > 0.0 and sentence_scores[i] >= 0.5):
                    self.sentence_correct_binary += 1.0

                # Total number of sentence with predicted label 1 (non-default label) = TP + FP
                if sentence_scores[i] >= 0.5:
                    self.sentence_predicted += 1.0
                # Total number of sentences with true label 1 (non-default label) = TP + FN
                if count_interesting_labels > 0.0:
                    self.sentence_total += 1.0
                # Total number of sentences with matching true and predicted labels (TP)
                if count_interesting_labels > 0.0 and sentence_scores[i] >= 0.5:
                    self.sentence_correct += 1.0
                # For each word (token), get its score
                for k in range(len(token_scores)):
                    self.append_token_data_for_sentence(k, true_labels, token_scores[k][i])

    def get_results(self, name):
        p = (float(self.sentence_correct) / float(self.sentence_predicted)) if (self.sentence_predicted > 0.0) else 0.0
        r = (float(self.sentence_correct) / float(self.sentence_total)) if (self.sentence_total > 0.0) else 0.0
        f = (2.0 * p * r / (p + r)) if ((p + r) > 0.0) else 0.0
        f05 = ((1.0 + 0.5 * 0.5) * p * r / ((0.5 * 0.5 * p) + r)) if (((0.5 * 0.5 * p) + r) > 0.0) else 0.0

        results = collections.OrderedDict()
        results[name + "_cost_sum"] = self.cost_sum
        results[name + "_cost_avg"] = self.cost_sum / float(self.sentence_count)
        results[name + "_sent_count"] = self.sentence_count
        results[name + "_sent_predicted"] = self.sentence_predicted
        results[name + "_sent_correct"] = self.sentence_correct
        results[name + "_sent_total"] = self.sentence_total
        results[name + "_sent_p"] = p
        results[name + "_sent_r"] = r
        results[name + "_sent_f"] = f
        results[name + "_sent_f05"] = f05
        results[name + "_sent_correct_binary"] = self.sentence_correct_binary
        results[name + "_sent_accuracy_binary"] = self.sentence_correct_binary / float(self.sentence_count)

        for k in range(len(self.token_ap_sum)):
            # Only calculate MAP over sentences that have any positive tokens
            mean_ap = self.token_ap_sum[k] / self.sentence_total
            p = (float(self.token_correct[k]) / float(self.token_predicted[k])) if (
                self.token_predicted[k] > 0.0) else 0.0
            r = (float(self.token_correct[k]) / float(self.token_total[k])) if (
                self.token_total[k] > 0.0) else 0.0
            f = (2.0 * p * r / (p + r)) if ((p + r) > 0.0) else 0.0
            f05 = ((1.0 + 0.5 * 0.5) * p * r / ((0.5 * 0.5 * p) + r)) if (
                ((0.5*0.5 * p) + r) > 0.0) else 0.0

            results[name + "_tok_" + str(k) + "_map"] = mean_ap
            results[name + "_tok_" + str(k) + "_p"] = p
            results[name + "_tok_" + str(k) + "_r"] = r
            results[name + "_tok_" + str(k) + "_f"] = f
            results[name + "_tok_" + str(k) + "_f05"] = f05

        results[name + "_time"] = float(time.time()) - float(self.start_time)

        return results

    def get_scikit_results(self):
        print("***********************************************************")
        print("Sentence pred: ")
        print(classification_report(self.sent_true, self.sent_pred))
        print("***********************************************************")
        print("Token pred: ")
        print(classification_report(self.tok_true, self.tok_pred))


class MLTMultiLabelEvaluator(object):
    def __init__(self, config, sent_label2id=None, token_label2id=None):
        self.config = config
        self.sent_label2id = sent_label2id
        self.sent_id2label = {v: k for k, v in self.sent_label2id.items()}
        self.token_label2id = token_label2id
        self.token_id2label = {v: k for k, v in self.token_label2id.items()}

        self.sent_true = []
        self.sent_pred = []
        self.tok_true = []
        self.tok_pred = []

        self.cost_sum = 0.0
        self.sentence_count = 0.0
        self.sentence_correct_binary = 0.0
        self.sentence_predicted = {k: 0.0 for k in self.sent_id2label.keys()}
        self.sentence_correct = {k: 0.0 for k in self.sent_id2label.keys()}
        self.sentence_total = {k: 0.0 for k in self.sent_id2label.keys()}

        self.token_ap_sum = {k: [] for k in self.token_id2label.keys()}
        self.token_predicted = {k: [] for k in self.token_id2label.keys()}
        self.token_correct = {k: [] for k in self.token_id2label.keys()}
        self.token_total = {k: [] for k in self.token_id2label.keys()}

        self.start_time = time.time()

    def calculate_ap(self, true_labels, predicted_scores, label):
        assert(len(true_labels) == len(predicted_scores))

        if self.config["debug_mode"]:
            with open("my_output.txt", "a") as f:
                f.write("\nTrue labels: \n" + str(" ".join([str(t) for t in true_labels])))
                f.write("\nPredicted labels: \n" + str(" ".join([str(t) for t in predicted_scores])))

        # This orders the predicted scores and returns the indices, in reverse order (biggest to smallest)
        # Here: but why do we do this?
        indices = numpy.argsort(numpy.array(predicted_scores))[::-1]

        summed, correct, total = 0.0, 0.0, 0.0
        for index in indices:
            total += 1.0
            if true_labels[index] == label:
                correct += 1.0
                summed += correct / total
        return (summed / correct) if correct > 0.0 else 0.0

    def append_token_data_for_sentence(self, index, true_labels, token_scores):
        # print("True token labels: ", true_labels)
        # print("Prediction token labels: ", token_scores)

        for key in self.token_id2label.keys():
            if len(self.token_ap_sum[key]) <= index:
                self.token_ap_sum[key].append(0.0)
                self.token_predicted[key].append(0.0)
                self.token_correct[key].append(0.0)
                self.token_total[key].append(0.0)

        for key in self.token_id2label.keys():
            ap = self.calculate_ap(true_labels, token_scores[:len(true_labels)], key)
            self.token_ap_sum[key][index] += ap

        # For each token in the sentence, calculate the same metrics as for the sentence scores
        for i in range(len(true_labels)):

            self.tok_true.append(true_labels[i])
            self.tok_pred.append(int(token_scores[i]))

            # Total number of tokens with predicted label 1 (non-default label) = TP + FP
            self.token_predicted[token_scores[i]][index] += 1.0

            # Total number of tokens with true label 1 (non-default label) = TP + FN
            self.token_total[true_labels[i]][index] += 1.0

            # Total number of tokens with both predicted and true labels 1 (non-default label) = TP
            if true_labels[i] == token_scores[i]:
                self.token_correct[true_labels[i]][index] += 1.0

    def append_data(self, cost, batch, sentence_scores, token_scores):
        for key in self.token_id2label.keys():
            assert (len(self.token_ap_sum[key]) == 0 or len(self.token_ap_sum[key]) == len(token_scores))
        self.cost_sum += cost

        for i in range(len(batch)):
            self.sentence_count += 1.0

            # HERE: this is the same as the labels in model, why not re-use so no mistakes made
            token_true_labels = [self.token_label2id[batch[i][j][-1]] for j in range(len(batch[i]) - 1)]
            sentence_true_label = self.sent_label2id[batch[i][-1][-1]]

            self.sent_true.append(sentence_true_label)
            self.sent_pred.append(int(sentence_scores[i]))

            # Total number of sentence-labels for which the correct label was assigned (accuracy)
            if sentence_true_label == sentence_scores[i]:
                self.sentence_correct_binary += 1.0

            # Total number of sentences with this predicted label = TP + FP
            self.sentence_predicted[sentence_scores[i]] += 1.0

            # Total number of sentences with this true label = TP + FN
            self.sentence_total[sentence_true_label] += 1.0

            # Total number of sentences with matching true and predicted labels (TP) for this specific label
            if sentence_true_label == sentence_scores[i]:
                self.sentence_correct[sentence_true_label] += 1.0

            # For each word (token), get its score
            for k in range(len(token_scores)):
                self.append_token_data_for_sentence(k, token_true_labels, token_scores[k][i])

    def get_results(self, name):

        results = collections.OrderedDict()
        results[name + "_cost_sum"] = self.cost_sum
        results[name + "_cost_avg"] = self.cost_sum / float(self.sentence_count)
        results[name + "_sent_count"] = self.sentence_count

        f_sent_macro = 0.0
        p_sent_macro = 0.0
        r_sent_macro = 0.0

        for key in self.sent_id2label.keys():
            p = (float(self.sentence_correct[key]) / float(self.sentence_predicted[key])) if (
                self.sentence_predicted[key] > 0.0) else 0.0
            r = (float(self.sentence_correct[key]) / float(self.sentence_total[key])) if (
                self.sentence_total[key] > 0.0) else 0.0
            f = (2.0 * p * r / (p + r)) if ((p + r) > 0.0) else 0.0
            f05 = ((1.0 + 0.5 * 0.5) * p * r / ((0.5 * 0.5 * p) + r)) if (((0.5 * 0.5 * p) + r) > 0.0) else 0.0
            key_name = "_key=%s" % self.sent_id2label[key]
            results[name + key_name + "_sent_predicted"] = self.sentence_predicted[key]
            results[name + key_name + "_sent_correct"] = self.sentence_correct[key]
            results[name + key_name + "_sent_total"] = self.sentence_total[key]
            results[name + key_name + "_sent_p"] = p
            results[name + key_name + "_sent_r"] = r
            results[name + key_name + "_sent_f"] = f
            p_sent_macro += p
            r_sent_macro += r
            f_sent_macro += f
            results[name + key_name + "_sent_f05"] = f05

        p_sent_macro /= len(self.sent_id2label.keys())
        r_sent_macro /= len(self.sent_id2label.keys())
        f_sent_macro /= len(self.sent_id2label.keys())

        p_sent_micro = (float(sum(self.sentence_correct.values())) / float(sum(self.sentence_predicted.values()))) if (
                sum(self.sentence_predicted.values()) > 0.0) else 0.0
        r_sent_micro =(float(sum(self.sentence_correct.values())) / float(sum(self.sentence_total.values()))) if (
                sum(self.sentence_total.values()) > 0.0) else 0.0
        f_sent_micro = (2.0 * p_sent_micro * r_sent_micro / (p_sent_micro + r_sent_micro)) if (
            (p_sent_micro + r_sent_micro) > 0.0) else 0.0

        results[name + "_sent_p_macro"] = p_sent_macro
        results[name + "_sent_r_macro"] = r_sent_macro
        results[name + "_sent_f_macro"] = f_sent_macro
        results[name + "_sent_p_micro"] = p_sent_micro
        results[name + "_sent_r_micro"] = r_sent_micro
        results[name + "_sent_f_micro"] = f_sent_micro

        results[name + "_sent_correct_binary"] = self.sentence_correct_binary
        results[name + "_sent_accuracy_binary"] = self.sentence_correct_binary / float(self.sentence_count)

        f_tok_macro = 0.0
        p_tok_macro = 0.0
        r_tok_macro = 0.0
        for key in self.token_id2label.keys():
            for k in range(len(self.token_ap_sum[key])):
                # Only calculate MAP over sentences that have any positive tokens
                mean_ap = self.token_ap_sum[key][k] / self.sentence_total[key]
                p = (float(self.token_correct[key][k]) / float(self.token_predicted[key][k])) if (
                    self.token_predicted[key][k] > 0.0) else 0.0
                r = (float(self.token_correct[key][k]) / float(self.token_total[key][k])) if (
                    self.token_total[key][k] > 0.0) else 0.0
                f = (2.0 * p * r / (p + r)) if ((p + r) > 0.0) else 0.0
                f05 = ((1.0 + 0.5 * 0.5) * p * r / ((0.5 * 0.5 * p) + r)) if (
                    ((0.5*0.5 * p) + r) > 0.0) else 0.0
                key_name = "_key=%s" % self.token_id2label[key]
                results[name + key_name + "_tok_" + str(k) + "_map"] = mean_ap
                results[name + key_name + "_tok_" + str(k) + "_p"] = p
                results[name + key_name + "_tok_" + str(k) + "_r"] = r
                results[name + key_name + "_tok_" + str(k) + "_f"] = f
                p_tok_macro += p
                r_tok_macro += r
                f_tok_macro += f
                results[name + key_name + "_tok_" + str(k) + "_f05"] = f05
        p_tok_macro /= len(self.token_id2label.keys())
        r_tok_macro /= len(self.token_id2label.keys())
        f_tok_macro /= len(self.token_id2label.keys())
        results[name + "_tok_p_macro"] = p_tok_macro
        results[name + "_tok_r_macro"] = r_tok_macro
        results[name + "_tok_f_macro"] = f_tok_macro

        p_tok_micro = (float(sum([self.token_correct[key][0] for key in self.token_correct.keys()])) / float(
            sum([self.token_predicted[key][0] for key in self.token_predicted.keys()]))) if (
            sum([self.token_predicted[key][0] for key in self.token_predicted.keys()]) > 0.0) else 0.0
        r_tok_micro = (float(sum([self.token_correct[key][0] for key in self.token_correct.keys()])) / float(
            sum([self.token_total[key][0] for key in self.token_total.keys()]))) if (
            sum([self.token_total[key][0] for key in self.token_total.keys()]) > 0.0) else 0.0
        f_tok_micro = (2.0 * p_tok_micro * r_tok_micro / (p_tok_micro + r_tok_micro)) if (
            (p_tok_micro + r_tok_micro) > 0.0) else 0.0
        results[name + "_tok_p_micro"] = p_tok_micro
        results[name + "_tok_r_micro"] = r_tok_micro
        results[name + "_tok_f_micro"] = f_tok_micro

        results[name + "_time"] = float(time.time()) - float(self.start_time)
        return results

    def get_scikit_results(self):
        print("***********************************************************")
        print("Sentence pred: ")
        print(classification_report(self.sent_true, self.sent_pred))
        f1_macro_sent = f1_score(self.sent_true, self.sent_pred, average="macro")
        f1_micro_sent = f1_score(self.sent_true, self.sent_pred, average="micro")
        print("F1-macro sent: ", f1_macro_sent)
        print("F1-micro sent: ", f1_micro_sent)

        print("***********************************************************")
        print("Token pred: ")
        print(classification_report(self.tok_true, self.tok_pred))
        f1_macro_tok = f1_score(self.tok_true, self.tok_pred, average="macro")
        f1_micro_tok = f1_score(self.tok_true, self.tok_pred, average="micro")
        print("F1-macro tok: ", f1_macro_tok)
        print("F1-micro tok: ", f1_micro_tok)
        print("***********************************************************\n")
