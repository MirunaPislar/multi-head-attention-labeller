from collections import OrderedDict
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import time


class Evaluator:
    def __init__(self, config, label2id_sent, label2id_tok):
        self.config = config
        self.id2label_sent = {v: k for k, v in label2id_sent.items()}
        self.id2label_tok = {v: k for k, v in label2id_tok.items()}

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

    def append_token_data_for_sentence(self, true_labels_tok, pred_labels_tok):
        # print("True token labels: ", true_labels_tok)
        # print("Pred token labels: ", pred_labels_tok)
        self.count_tok += len(true_labels_tok)

        # For each token, calculate the same metrics as for the sentence scores
        for true_label, pred_label in zip(true_labels_tok, pred_labels_tok):
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

    def append_data(self, cost, batch, sentence_scores, token_scores):
        self.cost_sum += cost
        self.count_sent += len(batch)

        for i, sentence in enumerate(batch):
            true_labels_tok = [token.label_tok for token in sentence.tokens]
            true_labels_sent = sentence.label_sent
            self.true_sent.append(true_labels_sent)
            self.pred_sent.append(sentence_scores[i])

            # Calculate accuracy.
            if true_labels_sent == sentence_scores[i]:
                self.correct_binary_sent += 1.0

            # Calculate TP + FP.
            self.sentence_predicted[sentence_scores[i]] += 1.0

            # Calculate TP + FN.
            self.sentence_total[true_labels_sent] += 1.0

            # Calculate TP.
            if true_labels_sent == sentence_scores[i]:
                self.sentence_correct[true_labels_sent] += 1.0

            # Get the scores for the tokens in this sentence
            self.append_token_data_for_sentence(true_labels_tok, list(token_scores[i])[:len(true_labels_tok)])

    @staticmethod
    def calculate_metrics(correct, predicted, total):
        p = correct / predicted if predicted > 0.0 else 0.0
        r = correct / total if total > 0.0 else 0.0
        f = 2.0 * p * r / (p + r) if (p + r) > 0.0 else 0.0
        f05 = (1 + 0.5 * 0.5) * p * r / (0.5 * 0.5 * p + r) if 0.5 * 0.5 * p + r > 0.0 else 0.0
        return p, r, f, f05

    def get_results(self, name):

        results = OrderedDict()
        results[name + "_cost_sum"] = self.cost_sum
        results[name + "_cost_avg"] = self.cost_sum / float(self.count_sent)

        results[name + "_count_sent"] = self.count_sent
        results[name + "_total_correct_sent"] = self.correct_binary_sent
        results[name + "_accuracy_sent"] = self.correct_binary_sent / float(self.count_sent)

        results[name + "_count_tok"] = self.count_tok
        results[name + "_total_correct_tok"] = self.correct_binary_tok
        results[name + "_accuracy_tok"] = self.correct_binary_tok / float(self.count_tok)

        # Calculate the micro and macro averages for the sentence predictions
        f_sent_macro, p_sent_macro, r_sent_macro = 0.0, 0.0, 0.0

        for key in self.id2label_sent.keys():
            p, r, f, f05 = self.calculate_metrics(
                self.sentence_correct[key], self.sentence_predicted[key], self.sentence_total[key])
            key_name = name + "_label=%s" % self.id2label_sent[key]
            # results[key_name + "_predicted_sent"] = self.sentence_predicted[key]
            # results[key_name + "_correct_sent"] = self.sentence_correct[key]
            # results[key_name + "_total_sent"] = self.sentence_total[key]
            results[key_name + "_precision_sent"] = p
            results[key_name + "_recall_sent"] = r
            results[key_name + "_f-score_sent"] = f
            # results[key_name + "_sent_f05"] = f05
            p_sent_macro += p
            r_sent_macro += r
            f_sent_macro += f

        p_sent_macro /= len(self.id2label_sent.keys())
        r_sent_macro /= len(self.id2label_sent.keys())
        f_sent_macro /= len(self.id2label_sent.keys())

        p_sent_micro, r_sent_micro, f_sent_micro, _ = self.calculate_metrics(
            sum(self.sentence_correct.values()),
            sum(self.sentence_predicted.values()),
            sum(self.sentence_total.values()))

        results[name + "_precision_macro_sent"] = p_sent_macro
        results[name + "_recall_macro_sent"] = r_sent_macro
        results[name + "_f-score_macro_sent"] = f_sent_macro
        results[name + "_precision_micro_sent"] = p_sent_micro
        results[name + "_recall_micro_sent"] = r_sent_micro
        results[name + "_f-score_micro_sent"] = f_sent_micro

        # Calculate the micro and macro averages for the token predictions
        f_tok_macro, p_tok_macro, r_tok_macro = 0.0, 0.0, 0.0
        for key in self.id2label_tok.keys():
            p, r, f, f05 = self.calculate_metrics(
                self.token_correct[key], self.token_predicted[key], self.token_total[key])
            key_name = name + "_label=%s" % self.id2label_tok[key]
            # results[key_name + "_predicted_tok"] = self.token_predicted[key]
            # results[key_name + "_correct_tok"] = self.token_correct[key]
            # results[key_name + "_total_tok"] = self.token_total[key]
            results[key_name + "_precision_tok"] = p
            results[key_name + "_recall_tok"] = r
            results[key_name + "_f-score_tok"] = f
            # results[key_name + "_tok_f05"] = f05
            p_tok_macro += p
            r_tok_macro += r
            f_tok_macro += f

        p_tok_macro /= len(self.id2label_tok.keys())
        r_tok_macro /= len(self.id2label_tok.keys())
        f_tok_macro /= len(self.id2label_tok.keys())

        p_tok_micro, r_tok_micro, f_tok_micro, _ = self.calculate_metrics(
            sum(self.token_correct.values()),
            sum(self.token_predicted.values()),
            sum(self.token_total.values()))

        results[name + "_precision_macro_tok"] = p_tok_macro
        results[name + "_recall_macro_tok"] = r_tok_macro
        results[name + "_f-score_macro_tok"] = f_tok_macro
        results[name + "_precision_micro_tok"] = p_tok_micro
        results[name + "_recall_micro_tok"] = r_tok_micro
        results[name + "_f-score_micro_tok"] = f_tok_micro

        results[name + "_time"] = float(time.time()) - float(self.start_time)
        return results

    def get_scikit_results(self):
        print("*" * 50)
        print("Sentence pred: ")
        print(classification_report(self.true_sent, self.pred_sent))
        f1_macro_sent = f1_score(self.true_sent, self.pred_sent, average="macro")
        f1_micro_sent = f1_score(self.true_sent, self.pred_sent, average="micro")
        print("F1-macro sent: ", f1_macro_sent)
        print("F1-micro sent: ", f1_micro_sent)

        print("*" * 50)
        print("Token pred: ")
        print(classification_report(self.true_tok, self.pred_tok))
        f1_macro_tok = f1_score(self.true_tok, self.pred_tok, average="macro")
        f1_micro_tok = f1_score(self.true_tok, self.pred_tok, average="micro")
        print("F1-macro tok: ", f1_macro_tok)
        print("F1-micro tok: ", f1_micro_tok)
        print("*" * 50)

