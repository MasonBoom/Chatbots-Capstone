import numpy as np
import re
import pickle
from chat_corpus_prep import pairs

# Building empty lists to hold sentences
input_docs = []
target_docs = []
# Building empty vocabulary sets
input_tokens = set()
target_tokens = set()

for line in pairs[:-1]:
    # Input and target sentences are separated by tabs
    input_doc, target_doc = line[0], line[1]
    # Appending each input sentence to input_docs
    input_docs.append(input_doc)
    # Splitting words from punctuation
    target_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", target_doc))
    # Redefine target_doc below
    # and append it to target_docs:
    target_doc = '<START> ' + target_doc + ' <END>'
    target_docs.append(target_doc)

    # Now we split up each sentence into words
    # and add each unique word to our vocabulary set
    for token in re.findall(r"[\w']+|[^\s\w]", input_doc):
        # Add your code here:
        if token not in input_tokens:
            input_tokens.add(token)
    for token in target_doc.split():
        # And here:
        if token not in target_tokens:
            target_tokens.add(token)

input_tokens = sorted(list(input_tokens))
target_tokens = sorted(list(target_tokens))

# Create num_encoder_tokens and num_decoder_tokens:
num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)

max_encoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", input_doc)) for input_doc in input_docs])
max_decoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", target_doc)) for target_doc in target_docs])

input_features_dict = dict(
    [(token, i) for i, token in enumerate(input_tokens)])
target_features_dict = dict(
    [(token, i) for i, token in enumerate(target_tokens)])

reverse_input_features_dict = dict(
    (i, token) for token, i in input_features_dict.items())
reverse_target_features_dict = dict(
    (i, token) for token, i in target_features_dict.items())

encoder_input_data = np.zeros(
    (len(input_docs), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for line, (input_doc, target_doc) in enumerate(zip(input_docs, target_docs)):

    for timestep, token in enumerate(re.findall(r"[\w']+|[^\s\w]", input_doc)):
        # Assign 1. for the current line, timestep, & word
        # in encoder_input_data:
        encoder_input_data[line, timestep, input_features_dict[token]] = 1.
        # add in conditional for handling unknown tokens (when token is not in input features dict)

    for timestep, token in enumerate(target_doc.split()):

        decoder_input_data[line, timestep, target_features_dict[token]] = 1.
        if timestep > 0:
            decoder_target_data[line, timestep - 1, target_features_dict[token]] = 1.