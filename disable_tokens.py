import random
random.seed(100)


def add_another_column(dataset, extension):
    """
    The original dataset file has multiple columns, the first one being
    the token and the last one the label. This method builds another file
    containing these as well as an additional middle column, representing
    the supervision status of a certain token, which can be "on" or "off".
    To start, all tokens are disabled. Later, we gradually increase the
    proportion of token-annotated sentences for which supervision is on.
    """
    path2 = dataset + ".column_added" + extension
    num_sent = 0
    with open(dataset + extension) as read_file, open(path2, 'w') as write_file:
        for line_tok_orig in read_file:
            line_tok = line_tok_orig.strip()
            if len(line_tok) == 0:
                num_sent += 1
                write_file.write(line_tok_orig)
                continue
            line_tok = line_tok.split()
            write_file.write(line_tok[0] + "\t" + "off" + "\t" + line_tok[-1] + "\n")
    return path2, num_sent


def convert_labels(read_dataset, write_dataset, percent, no_sentences_to_enable):
    """
    Takes a file and randomly enables no_sentences_to_enable of them.
    :param read_dataset: file to read from
    :param write_dataset: file to write to
    :param percent: percent to enable (used to calculate the probability of enabling)
    :param no_sentences_to_enable: how many sentences should be enabled
    :return: sentences enabled
    """
    sentences_enabled = 0
    write_file_str = ""
    with open(read_dataset) as read_file:
        lines = read_file.read().split("\n")

    line_index = 0
    while line_index < len(lines):
        line = lines[line_index].strip()
        if len(line) == 0:
            write_file_str += "\n"
            line_index += 1
            continue

        line_tok = line.split()
        assert len(line_tok) > 2, "Line tok shouldn't be empty!"
        prob = random.random()
        conf = (percent + 3) / 100.0

        if conf < 1 and (prob > conf or line_tok[1] == "on"
                         or sentences_enabled >= no_sentences_to_enable):
            while len(line) != 0:
                write_file_str += line + "\n"
                line_index += 1
                line = lines[line_index].strip()
        else:
            sentences_enabled += 1
            while len(line) != 0:
                line_tok = line.split()
                write_file_str += (line_tok[0] + "\t" + "on" + "\t"
                                   + line_tok[-1] + "\t" + "\n")
                line_index += 1
                line = lines[line_index].strip()
        write_file_str += "\n"
        line_index += 1

    with open(write_dataset, 'w') as write_file:
        write_file.write(write_file_str)
    return sentences_enabled


filename = "../data/fce_semi_supervised/fce.train"
ext = ".tsv"
curr_filename, no_sentences = add_another_column(filename, ext)
print("No. sentences = ", no_sentences)
pace = 10

for i in range(10, 110, pace):
    wanted_enabled = int((pace / 100) * no_sentences)
    start = i
    prev_filename = curr_filename
    curr_filename = filename + "_%d_percent" % i + ext
    actually_enabled = convert_labels(
        prev_filename, curr_filename, i, wanted_enabled)
    print("Current filename is %s. Percent = %.1f. We want to enable %d."
          % (curr_filename, i, wanted_enabled))
    while actually_enabled < wanted_enabled:
        wanted_enabled -= actually_enabled
        print("Only got ", actually_enabled, "need ", wanted_enabled, " more...")
        actually_enabled = convert_labels(
            curr_filename, curr_filename, i, wanted_enabled)
        print(" so we added ", actually_enabled, " more!")

