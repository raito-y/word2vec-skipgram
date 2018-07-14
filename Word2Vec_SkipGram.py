# Vanilla Word2Vec SkipGram model using TensorFlow.
# Simplified version of original code available on tensorflow github.


import numpy as np
import tensorflow as tf

import os
import sys
import argparse
import collections
import zipfile
import random
import math
from tempfile import gettempdir
from six.moves import urllib
from six.moves import xrange
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.contrib.tensorboard.plugins import projector


# Provide directory path to save TensorFlow summaries.
LOG_DIR = ""

# Provide URL for the data file.
FILE_URL = "http://mattmahoney.net/dc/text8.zip"


# Download the file if not present.
def download():
	url = FILE_URL
	filename = url[url.rfind("/")+1:]
	local_filename = os.path.join(gettempdir(), filename)
	if not os.path.exists(local_filename):
        local_filename, _ = urllib.request.urlretrieve(FILE_URL, local_filename)
    return local_filename


filename = download()


# Read the data into a list of strings.
def read_data(filename, filetype):
	if filetype == "zip":
		with zipfile.ZipFile(filename) as f:
			data = tf.compat.as_str(f.read(f.namelist()[0])).split()
	elif filetype == "text":
		with open(filename) as f:
			data = tf.compat.as_str(f.read()).split()
	return data


vocabulary = read_data(filename, "zip")


# Build the dictionary and replace all rare words with UNK token.
vocabulary_size = 50000

def build_dataset(words, n_words):
    
    # data - list of codes (integers from 0 to vocabulary_size-1). Original text but words are replaced by their codes
    # count - map of words(strings) to count of occurrences
    # dictionary - map of words(strings) to their codes(integers)
    # reverse_dictionary - maps codes(integers) to words(strings)
    
    # Add 'UNK' at top of the list as single word for less frequency words. Frequency will be updated later.
    count = [['UNK', -1]]
    # Counter creates a list of frequent words and their frequency. E.g. [['UNK', -1], ('abc', 3), ('jkl', 2)]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    
    dictionary = dict()
    # Create a dictionary of frequent words and assign index values to the words.
    for word, _ in count:
        dictionary[word] = len(dictionary)
    
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            # Count less frequent ('UNK') words
            unk_count += 1
        # Store dictionary indexes for all words from the 'words' list
        data.append(index)
    
    # Update frequency of 'UNK' word added at the top
    count[0][1] = unk_count
    
    # Create reverse dictionary where indexes are keys and corresponding words are values
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    
    return data, count, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)


del vocabulary  # To reduce memory.


# Counter over the text being processed
data_index = 0


# Function to generate a training batch for skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    
    global data_index
    
    assert batch_size % num_skips == 0
    # num_skips less than or equal to context size. Number of samples to be picked is less than or equal to the context size.
    assert num_skips <= 2 * skip_window
    
    # Input word vector
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    # Output / context words vector
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    
    # dqueue used to hold text of window size 'span'.
    # dqueue is list-like container with fast appends and pops on either end.
    buffer = collections.deque(maxlen=span)
    
    # Reset the counter when all words from the text are processed.
    if data_index  + span > len(data):
        data_index = 0
        
    # Fetch the next set of words.
    buffer.extend(data[data_index:data_index + span])
    
    # Update the data_index position.
    data_index += span
    
    for i in range(batch_size // num_skips):

        # Fetch buffer positions of context words except the center word E.g. [0, 1, 3, 4] for span = 5. 
        context_words = [w for w in range(span) if w != skip_window]
        
        # Sample words from context_words
        words_to_use = random.sample(context_words, num_skips)
        
        # Loop over all the context words in this window. They have the same center word.
        for j, context_word in enumerate(words_to_use):
            
            # Add the center word to input word vector for this context.
            batch[i * num_skips + j] = buffer[skip_window]
            
            print(batch[i * num_skips + j])
            
            # Add the context word to context words vector.
            labels[i * num_skips + j] = buffer[context_word]
            
            if data_index == len(data):
                buffer.extend(data[0:span])
                data_index = 0
            else:
                # Move to the next window by 1.
                buffer.append(data[data_index])
                data_index += 1
                
    # Backtrack a little to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
        
    return batch, labels


batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)  # Three words window
# Print all pairs of i/p and o/p words with their indexes

for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], "->", labels[i, 0], reverse_dictionary[labels[i, 0]])


# Build and train a skip-gram model.

batch_size = 128
embedding_size = 128 # Features of embedding vector.
skip_window = 1 # How many words to consider left and right.
num_skips = 2 # How many times to reuse an input to generate a label or how many words from context to consider as label.
num_sampled = 64 # Number of negative examples to sample. 

# These 3 variables are used only for displaying model accuracy, they don't affect calculations.
# The purpose is to find words which are symentically similar to a word from a random sample.
# We generate a random validation set to sample nearest neighbors.
# Here we have limited the validation the samples to the words which are most frequent.
# Pick random 16 words among 100 words from top of a distribution to evaluate similarity on.
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

graph = tf.Graph()

with graph.as_default():
    
    # Input data
    with tf.name_scope("inputs"):
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        
    with tf.device("/device:GPU:0"):
        # Look up embeddings for inputs.
        with tf.name_scope("embeddings"):
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
            
    # Construct the variables for the NCE loss.
    with tf.name_scope("weights"):
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0/math.sqrt(embedding_size)))
    with tf.name_scope("biases"):
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        
    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each time we evaluate the loss.
    # Explanation of the meaning of NCE loss:
    # http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
    print(nce_weights.dtype)
    print(train_inputs.dtype)
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=num_sampled,
                num_classes=vocabulary_size))
        
    # Add the loss value as a scalar to the summar.
    tf.summary.scalar("loss", loss)
    
    # Construct a SGD optimizer with a learning rate of 1.0.
    with tf.name_scope("optimizer"):
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        
    # Compute the cosine similarity between minibatch examples and all embeddings.
    # https://cmry.github.io/notes/euclidean-v-cosine
    # This comes out to be same as (∑A.∑B) / sqrt(∑(A2)sqrt(∑(B2)
    # In this case sqrt(∑(A2) is used as denominator.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    # Valid set is also derived from the normalized set.
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)

    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    
    # Merge all summaries.
    merged = tf.summary.merge_all()
    
    # Add variable initializer.
    init = tf.global_variables_initializer()
    
    # Create a saver
    saver = tf.train.Saver()


# Begin training.

num_steps = 100001

with tf.Session(graph=graph) as session:
    # Open a writer to write summaries.
    writer = tf.summary.FileWriter(LOG_DIR, session.graph)
    
    # Initalize variables.
    init.run()
    print("Initialized.")
    
    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        
        # Define metadata variable.
        run_metadata = tf.RunMetadata()
        
        # Run operations and evaluate tensors in featches.
        # Returns list as per fetches.
        # Feed metadata variable to session for visualizing the graph in TensorBoard.
        _, summary, loss_val = session.run(
            fetches=[optimizer, merged, loss],
            feed_dict=feed_dict,
            run_metadata=run_metadata)
        average_loss += loss_val
        
        # Add returned summaries to writer in each step with the step number.
        writer.add_summary(summary=summary, global_step=step)
        # Add metadata to visualize the graph for the last session.run().
        if step == (num_steps-1):
            writer.add_run_metadata(run_metadata, "step%d" % step)
            
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is the estimate of the loss of over last 200 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0
        
        # Used only for displaying model accuracy and doesn't affect calculations.
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8 # Number of nearest neighbors.
                nearest = (-sim[i, :]).argsort()[1:top_k+1]  # This provides the indexes of the nearest words.
                log_str = "Nearest to %s: " % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s, " % (log_str, close_word)
                print(log_str)
    
    final_embeddings = normalized_embeddings.eval()
    
    # Write the corresponding labels for the embeddings.
    with open(LOG_DIR + "/metadata.tsv", "w") as f:
        for i in xrange(vocabulary_size):
            f.write(reverse_dictionary[i] + "\n")
            
    # Save the model for checkpoints.
    saver.save(session, os.path.join(LOG_DIR, "model.ckpt"))

    # Create a configuration for visualizing embeddings with the labels in Tensorboard.
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.add()
    embedding_conf.metadata_path = os.path.join(LOG_DIR, "metadata.tsv")
    projector.visualize_embeddings(writer, config)
    
writer.close()


# Visualize the embeddings

# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=[18, 18]) # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords="offset points",
            ha="right",
            va="bottom")
    plt.savefig(filename)
    
tsne = TSNE(n_components=2, perplexity=30, init="pca", n_iter=5000, method="exact")
plot_only = 500
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [reverse_dictionary[i] for i in xrange(plot_only)]
plot_with_labels(low_dim_embs, labels, os.path.join(LOG_DIR, "tsne.png"))

# Original Code link: https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/examples/tutorials/word2vec/word2vec_basic.py