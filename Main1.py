<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.optim as optim

# Define the Encoder and Decoder models
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = nn.functional.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

# Function to train the model
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = torch.zeros(hidden_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]])

    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()

        loss += criterion(decoder_output, target_tensor[di])

        if decoder_input.item() == EOS_token:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

# Function to train the chatbot
def train_chatbot(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    optimizer_encoder = optim.SGD(encoder.parameters(), lr=learning_rate)
    optimizer_decoder = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        input_tensor, target_tensor = random_training_pair()
        loss = train(input_tensor, target_tensor, encoder, decoder, optimizer_encoder, optimizer_decoder, criterion)

        if iter % print_every == 0:
            print(f"Iteration {iter}, Loss: {loss}")

# Function to evaluate the chatbot
def evaluate(encoder, decoder, sentence, max_length = max):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = torch.zeros(hidden_size)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]])

        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words

# Example usage
hidden_size = 256
MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1

# Create encoder and decoder
encoder = Encoder(input_size, hidden_size)
decoder = Decoder(hidden_size, output_size)

# Train the chatbot
train_chatbot(encoder, decoder, n_iters=10000, print_every=1000)

# Test the chatbot
sentence = "How are you?"
output_words = evaluate(encoder, decoder, sentence)
output_sentence = ' '.join(output_words)
print(f"Input: {sentence}")
print(f"Output: {output_sentence}")
=======
import torch
import torch.nn as nn
import torch.optim as optim

# Define the Encoder and Decoder models
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size  # Store hidden_size as an attribute
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size  # Store hidden_size as an attribute
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = nn.functional.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = nn.functional.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

# Function to train the model
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = torch.zeros(1, 1, hidden_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]])

    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()

        loss += criterion(decoder_output, target_tensor[di])

        if decoder_input.item() == EOS_token:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

# Function to train the chatbot
def train_chatbot(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    optimizer_encoder = optim.SGD(encoder.parameters(), lr=learning_rate)
    optimizer_decoder = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        input_tensor, target_tensor = random_training_pair()
        loss = train(input_tensor, target_tensor, encoder, decoder, optimizer_encoder, optimizer_decoder, criterion)

        if iter % print_every == 0:
            print(f"Iteration {iter}, Loss: {loss}")

import random

# Example list of sentence pairs for training
pairs = [
    ("How are you", "I am fine"),
    ("Hello", "Hi there"),
    # Add more pairs here based on your dataset
]

def random_training_pair():
    # Select a random pair from the dataset
    pair = random.choice(pairs)
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return input_tensor, target_tensor


# Function to evaluate the chatbot
def evaluate(encoder, decoder, sentence, max_length = max):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = torch.zeros(1, 1, hidden_size)
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]])

        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words
    
def tensorFromSentence(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    indexes.append(EOS_token)  # Add EOS token to the end of the sentence
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)
    
class Language:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Start with 2 tokens (SOS and EOS)

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1



# Example usage
hidden_size = 256
input_size = 1000  # Replace 1000 with the actual vocabulary size
output_size = 1000  # Replace 1000 with the actual vocabulary size
MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1

# Define your language objects
input_lang = Language("English")
output_lang = Language("English")

# Populate vocabulary by adding each sentence in pairs
pairs = [
    ("How are you", "I am fine"),
    ("Hello", "Hi there"),
    # Add more pairs here
]

for pair in pairs:
    input_lang.add_sentence(pair[0])
    output_lang.add_sentence(pair[1])

# Now set input_size and output_size
input_size = input_lang.n_words
output_size = output_lang.n_words


# Create encoder and decoder
encoder = Encoder(input_size, hidden_size)
decoder = Decoder(hidden_size, output_size)

# Train the chatbot
train_chatbot(encoder, decoder, n_iters=10000, print_every=1000)

# Test the chatbot
sentence = "How are you?"
output_words = evaluate(encoder, decoder, sentence)
output_sentence = ' '.join(output_words)
# print(f"Input: {sentence}")
# print(f"Output: {output_sentence}")
print("Input Language Vocabulary:", input_lang.word2index)
print("Output Language Vocabulary:", output_lang.word2index)
>>>>>>> 71dd97b (Update to Chatbot)
