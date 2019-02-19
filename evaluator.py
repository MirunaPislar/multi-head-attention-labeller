import time
import collections
import numpy


class MLTEvaluator(object):
    def __init__(self, config):
        self.config = config
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
            # Total number of tokens with predicted label 1 (non-default label) = TP + FP
            if token_scores[i] >= 0.5:
                self.token_predicted[index] += 1.0
            # Total number of tokens with true label 1 (non-default label) = TP + FN
            if true_labels[i] >= 0.5:
                self.token_total[index] += 1.0
            # Total number of tokens with both predicted and true labels 1 (non-default label) = TP
            if true_labels[i] >= 0.5 and token_scores[i] >= 0.5:
                self.token_correct[index] += 1.0

    def append_data(self, cost, batch, sentence_scores, token_scores):
        assert (len(self.token_ap_sum) == 0 or len(self.token_ap_sum) == len(token_scores))
        self.cost_sum += cost

        for i in range(len(batch)):
            self.sentence_count += 1.0
            # HERE: this is the same as the labels in model, why not re-use so no mistakes made
            true_labels = [0.0 if batch[i][j][-1] == self.config["default_label"]
                           else 1.0 for j in range(len(batch[i]))]
            count_interesting_labels = numpy.array(true_labels).sum()

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
