import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from tqdm import tqdm
import numpy as np

mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14

html_header = '<!DOCTYPE html>\n<html>\n<font size="5">\n<head>\n<meta charset="UTF-8">\n<body>\n'
html_footer = '</body></font></html>'

# A couple of colours (expecting no more than 10 heads). Add more if needed.
head_colours = [[1.0, 0.0, 0.0],  # red
                [0.0, 1.0, 0.0],  # green
                [0.0, 0.0, 1.0],  # blue
                [1.0, 0.0, 1.0],  # pink
                [1.0, 0.6, 0.0],  # orange
                [0.3, 0.3, 0.3],  # grey
                [0.6, 0.0, 1.0],  # purple
                [1.0, 1.0, 0.3],  # yellow
                [0.0, 0.6, 1.0],  # a type of green
                [0.5, 1.0, 0.0],  # a type of blue
                ]
head_colours_sent = [[0.8, 0.0, 0.4], [0.0, 0.4, 0.4]]  # for binary-labelled sentences


def plot_token_scores(token_probs, sentence, plot_name=None, show=False):
    """
    Plot the (normalized) token scores onto a grid of heads.
    :param token_probs: normalized token scores of shape [batch_size, num_heads].
    :param sentence: contains all the tokens corresponding to the token probs.
    :param plot_name: name of file where to save the plot. Doesn't save it if None.
    :param show: whether to show or not the plot to the screen.
    :return: Nothing, just plot the token scores.
    """
    sentence_length = len(sentence.tokens)
    token_probs = token_probs[:][:sentence_length].T
    (nrows, ncols) = token_probs.shape
    color_data = []

    for i, [r, g, b] in enumerate(head_colours[:nrows]):
        row = []
        for j in range(ncols):
            row.append([r, g, b, token_probs[i][j]])
        color_data.append(row)

    plt.figure(figsize=(16, 12), dpi=100)
    row_labels = ["head %d" % i for i in range(nrows)]
    col_labels = [token.value for token in sentence.tokens]
    plt.imshow(color_data, vmin=0, vmax=sentence_length)
    plt.xticks(range(ncols), col_labels, rotation=45)
    plt.yticks(range(nrows), row_labels)
    plt.tight_layout()
    if plot_name is not None:
        plt.savefig("%s_%d.png" % (plot_name, int(time.time())), format="png", dpi=100)
    if show:
        plt.show()


def plot_predictions(
        all_sentences, all_sentence_probs, all_token_probs,
        html_file, sent_binary=False):
    """
    Writes a HTML file with the predictions at the sentence and token level.
    :param all_sentences: list of all the sentences in all batches.
    :param all_sentence_probs: a list of all the sentence probabilities in all batches;
    each batch of sentence_prob has shape [B, num_sent_labels] and must contain normalized data.
    :param all_token_probs: a list of all the token probabilities in all batches;
    each batch of token_probs has shape [B, M, num_tok_labels] and must contain normalized data.
    :param html_file: name of the html file that will be produced.
    :param sent_binary: whether the sentence labels are binary or not. This is needed
    to use different colours than the token labels if the sentence labels don't match
    the token labels (for our purposes, this happens when the sentence labels are binary).
    :return: Nothing, just saves a html file with the coloured predictions,
     which you can see in your browser.
    """
    print("Plotting predictions across all batches...")
    with open("%s_%d.html" % (html_file, int(time.time())), "w") as html_file:

        # Write the normal html file header.
        html_file.write(html_header)

        # Go through each batch.
        for sentences, sentence_probs, token_probs in tqdm(zip(
                all_sentences, all_sentence_probs, all_token_probs),
                total=len(all_sentences)):

            # Go through each sentence in the batch.
            for sent, sent_prob, tok_probs_this_sent in zip(
                    sentences, sentence_probs, token_probs):
                # Represent in colour the gold and the predicted sentence label.
                predicted_sent_label = int(np.argmax(sent_prob))
                gold_sent_label = sent.label_sent
                alpha_sent = sent_prob[predicted_sent_label]

                if sent_binary:
                    [r_pred, g_pred, b_pred] = head_colours_sent[predicted_sent_label]
                    [r_gold, g_gold, b_gold] = head_colours_sent[gold_sent_label]
                else:
                    [r_pred, g_pred, b_pred] = head_colours[predicted_sent_label]
                    [r_gold, g_gold, b_gold] = head_colours[gold_sent_label]

                html_file.write(
                    '<font style="background: rgba(%d, %d, %d, %f)">%s</font>\n'
                    % (int(r_pred * 255), int(g_pred * 255), int(b_pred * 255),
                       alpha_sent, "PRED=%s" % str(predicted_sent_label)))
                html_file.write(
                    '<font style="background: rgba(%d, %d, %d, %f)">%s</font>\n'
                    % (int(r_gold * 255), int(g_gold * 255), int(b_gold * 255),
                       0.9, "GOLD=%s" % str(gold_sent_label)))

                # Write each token in the colour background of its most probable
                #  head prediction. Incorrect predictions will be underlined.
                for token, tok_prob in zip(sent.tokens, tok_probs_this_sent):
                    predicted_head = int(np.argmax(np.argmax(tok_prob)))
                    alpha_tok = tok_prob[predicted_head]
                    [r, g, b] = head_colours[predicted_head]
                    if predicted_head == token.label_tok:
                        token_html = "%s" % token.value
                    else:
                        token_html = "<u>%s<u>" % token.value
                    html_file.write(
                        '<font style="background: rgba(%d, %d, %d, %f)">%s</font>\n'
                        % (int(r * 255), int(g * 255), int(b * 255),
                           alpha_tok, token_html))
                html_file.write('<br><br>')
        html_file.write(html_footer)
    print("HTML visualizations: Done!")
