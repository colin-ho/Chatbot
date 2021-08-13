# Chatbot
AI and pattern-based chatbot made with Tensorflow for personal portfolio site

## Overview
This chatbot was made to answer professional questions about myself, and to enhance user experience on my personal website. It is an intent-based chatbot built with a neural network for text classification using 50 dimensional GLOVE word embeddings, a 1-D convolutional layer, and LSTM layer. 


## Data
The data used to train this chatbot is in the intents.json file. It is a list of possible intents a user might have when using the chatbot, and a list of potential patterns of words a user might use to express that intent. Simply put, the patterns are the data and the intent is the label.


## Pre-processing
There are several steps for text-preprocessing.

1. Vocabulary creation. Create a vocab list comprising of all the root words in the data. This is necessary later on to create the encoded data and embedding matrix.
2. Lemmatization and stop-word removal. This process removes any stopwords (commonly used words that don't contribute to the meaning of the sentence) and lemmatizes the words to their root. This is to minimize the size of the vocab list and make a more concise model.
3. Tokenization and encoding. Convert the sentences into vectors, where each word will be converted into a number corresponding to their position in the vocab list. The result is an encoded matrix of all the tokenized sentences. The vecotrized sentences will be padded according to the length of the longest sentence, which in this case is 9.


## Word Embedding
The word embedding layer converts each word into a vector. Passing the encoded matrix through this layer results in a three-dimensional tensor of shape (number of sentences, length of longest sentence, dimension of embedding). This word embedding layer is useful because it represents the similarity between words as the closeness between their vectors, thus allowing the model to better identify actual linguistic meaning and context. 

The embedding layer is created by collecting a pre-trained vector from the GLOVE 50D embeddings that corresponds to each word in the vocab list. Therefore the shape of the embedding layer would be (vocab size, 50).


## Model Training
Training the model through a neural network architecture is done immediately after passing the encoded data into the embedding matrix. The layers used in this NN are a 1-D convolutional layer, max pooling layer, birectional LSTM layer, and a 13 dimensional dense layer with softmax activation. 

1. Convolutional and max pooling layer. The 1-D convolutional layer uses filters to extract features from sentences. Each filter will slide through the sentence with 3 words per filter (since the kernel size is 3) to extract a feature from those 3 words. Since our sentence length is 9, and padding is set to 'same', a total of 9 features will be extracted from each filter. A filter quantity of 50 will mean that a total of 50 features will be extracted. The resultant matrix for a sentence after passing through the convolutional layer is (9,50). The max pooling layer will then take the 3 most important features from these 9 and return a (3,50) matrix. In essence, the convolutional part of this model extracts the most important features from a sentence.

2. LSTM layer. An LSTM layer is a neural network that takes in an input sequence and looks at it from left to right, formulating a hidden state based on the previous hidden states, without forgetting or minimizing the importance of those previous states. It is useful for textual data because meaning of text often depends on the words that come before it. The output of the LSTM would then be the hidden state of the last feature. The dimension of the output for this model is 50, it is small because the input data only has a length of 3. Essentially, the LSTM sequentially condenses the features of a sentence into a one-dimensional vector which can be processed into a dense layer with softmax activation to create predictions.

![Model Summary][model-summary]

#### Why I chose this model architecture:
Conventional text classification models usually use either a CNN based architecture or RNN based architecture. However, the dataset I am working with is small, limited to only 9 words, yet has 13 different possible classifications. I initially tried with a CNN based network. While it was fast, it was prone to overfitting, and techniques such as max pooling and dropout layers did not solve this issue. Perhaps it is due to the small dimensionality of the input data. The LSTM network was better in terms of validation accuracy because it has the benefit of sequential memory, however it was only reaching up to 75% validation accuracy. This could be due to the similarities between sentences in the input data, as many of them, even those with different labels, have many words in common. As such, sentences with completely different labels will have very similar outputs from the LSTM. Combining the two models solved the issue. The CNN layers extract the key features, which then are fed into the LSTM which accounts for their sequential context, and achieving test accuracy of 100%.


## Usage
To try out this chatbot, simply upload the json and pkl files to the chatbot.ipynb notebook. Then, scroll to the bottom where the model is loaded, activate the prediction functions, and test with your own input.

To train this chatbot for your own use, modify the intents.json file for your own purposes, then start the notebook from the top. Take note that several parameters will have to be manually modified, such as the max length of your sentences, before training the model.


## References
1. https://medium.com/@findsoulyourself/secrets-behind-the-convolutional-neural-networks-and-lstm-8ad338eaacfe
2. https://medium.com/@mrunal68/text-sentiments-classification-with-cnn-and-lstm-f92652bc29fd
3. https://medium.com/@audreyctang/an-intuitive-comparison-of-nlp-models-neural-networks-rnn-cnn-lstm-fc11bf452923

[product-screenshot1]: images/1.png