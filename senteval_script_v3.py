import sys
import senteval
import numpy as np
import torch
import torch.nn as nn
import os
import pickle
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

start_script = time.time()

# Set encoder type
encoder_type = 'bilstm-last'  # Options: 'bilstm-last', 'bilstm-max', 'gated-cnn'

# Simple BiLSTM Encoder
class BiLSTMEncoder(nn.Module):
    def __init__(self, word_emb_dim, hidden_dim=2048, pooling='last'):
        super(BiLSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.lstm = nn.LSTM(word_emb_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)

    def forward(self, sent_embeddings):
        lstm_out, _ = self.lstm(sent_embeddings)
        if self.pooling == 'last':
            return lstm_out[:, -1, :]
        elif self.pooling == 'max':
            return torch.max(lstm_out, dim=1).values
        else:
            raise ValueError("Invalid pooling type. Choose 'last' or 'max'.")

# Simple Gated ConvNet Encoder
class GatedConvNetEncoder(nn.Module):
    def __init__(self, word_emb_dim, hidden_dim=2048, kernel_size=3):
        super(GatedConvNetEncoder, self).__init__()
        self.conv = nn.Conv1d(word_emb_dim, hidden_dim, kernel_size=kernel_size, padding=1)
        self.gate = nn.Conv1d(word_emb_dim, hidden_dim, kernel_size=kernel_size, padding=1)

    def forward(self, sent_embeddings):
        conv_out = self.conv(sent_embeddings.transpose(1, 2))
        gate_out = torch.sigmoid(self.gate(sent_embeddings.transpose(1, 2)))
        gated_output = conv_out * gate_out
        return torch.max(gated_output, dim=2).values


def create_dictionary(sentences, threshold=0):
    # Check if cached dictionary exists
    if os.path.exists("cached_word2id.pkl"):
        with open("cached_word2id.pkl", "rb") as f:
            return None, pickle.load(f)
    
    # Create dictionary if not cached
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {word: count for word, count in words.items() if count >= threshold}
        words = newwords
    words['<s>'], words['</s>'], words['<p>'] = 1e9 + 4, 1e9 + 3, 1e9 + 2

    word2id = {word: i for i, (word, _) in enumerate(sorted(words.items(), key=lambda x: -x[1]))}
    
    # Cache dictionary
    with open("cached_word2id.pkl", "wb") as f:
        pickle.dump(word2id, f)
    
    return None, word2id

def get_wordvec(glove_path, word2id):
    start_get_wordvec = time.time()
    word_vec = {}
    loaded_count = 0
    
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            split_line = line.strip().split()
            word = split_line[0]
            if word in word2id:
                try:
                    vector = np.array(split_line[1:], dtype='float32')
                    word_vec[word] = vector
                    loaded_count += 1
                    
                    # Stop loading if we have all words
                    if loaded_count >= len(word2id):
                        break
                except ValueError:
                    print(f"Warning: Could not convert embedding for word '{word}'. Skipping this word.")
                    
    print("Total get_wordvec function time:", time.time() - start_get_wordvec)
    print(f"Loaded {len(word_vec)} out of {len(word2id)} requested word vectors.")
    return word_vec
"""
def simple_average_encoder(sent_tensor):
    return torch.mean(sent_tensor, dim=1)
"""
print("DEBUG: Starting prepare function")
# SentEval prepare function
def prepare(params, samples):
    start_prepare = time.time()
    
    
    _, params['word2id'] = create_dictionary(samples)
    params['word_vec'] = get_wordvec(r"C:\Users\mugeb\Internship_Project\SentEval-main\SentEval-main\data\glove.840B.300d.txt", params['word2id'])
    params['wvec_dim'] = 300

    hidden_dim = 512
    if encoder_type == 'bilstm-last':
        params['encoder'] = BiLSTMEncoder(word_emb_dim=params['wvec_dim'], pooling='last')
    elif encoder_type == 'bilstm-max':
        params['encoder'] = BiLSTMEncoder(word_emb_dim=params['wvec_dim'], pooling='max')
    elif encoder_type == 'gated-cnn':
        params['encoder'] = GatedConvNetEncoder(word_emb_dim=params['wvec_dim'])
    else:
        raise ValueError("Invalid encoder_type. Choose 'bilstm-last', 'bilstm-max', or 'gated-cnn'")

    print("Total prepare function time:", time.time() - start_prepare)
    return
print("DEBUG: Prepare function completed")

def batcher(params, batch):
    import time
    start_batcher = time.time()
    print("DEBUG: Starting batcher function")  # Debug start
    print(f"DEBUG: Batch size = {len(batch)}")  # Print batch size
    sentence_embeddings = []
    
    # Adjust batch size dynamically
    #batch = batch[:3]  # Start with a small number to test

    for idx, sentence in enumerate(batch):
        print(f"DEBUG: Processing sentence {idx + 1}: {sentence}")  # Debug individual sentences
        words = [word.lower() for word in (sentence if isinstance(sentence, list) else sentence.split())]
        
        # Retrieve word embeddings only for words that exist in word_vec
        word_embeddings = [torch.tensor(params['word_vec'].get(word, np.zeros(300)), dtype=torch.float32)
                           for word in words if word in params['word_vec']]
        
        if word_embeddings:
            sent_tensor = torch.stack(word_embeddings).unsqueeze(0)
            sentence_embedding = params['encoder'](sent_tensor).detach().numpy()
            sentence_embeddings.append(sentence_embedding)
        else:
            sentence_embeddings.append(np.zeros((1, params['wvec_dim'])))

    output = np.vstack(sentence_embeddings)
    print("Total batch processing time:", time.time() - start_batcher)
    print("DEBUG: Batch processing completed")  # Debug end
    return output

'''
def batcher(params, batch):
    sentence_embeddings = []
    for sentence in batch:
        word_embeddings = [torch.tensor(params['word_vec'].get(word, np.zeros(300)), dtype=torch.float32) for word in sentence]
        if word_embeddings:
            sent_tensor = torch.stack(word_embeddings).unsqueeze(0)
            sentence_embedding = simple_average_encoder(sent_tensor).detach().numpy()
            sentence_embeddings.append(sentence_embedding)
        else:
            sentence_embeddings.append(np.zeros((1, 300)))
    return np.vstack(sentence_embeddings)
'''
'''
# Batcher function
# Optimized batcher function
def batcher(params, batch):
    start_batcher = time.time()
    
    # Convert all sentences in the batch to tensors in one go
    batch_embeddings = []
    for sentence in batch:
        words = sentence if isinstance(sentence, list) else sentence.split()
        
        # Retrieve all word embeddings and stack them once
        word_embeddings = [params['word_vec'].get(word, np.zeros(300)) for word in words]
        
        # Convert the list of numpy arrays directly to a single tensor for the sentence
        if word_embeddings:
            sent_tensor = torch.tensor(word_embeddings, dtype=torch.float32).unsqueeze(0)
            
            # Process with encoder
            sentence_embedding = params['encoder'](sent_tensor)
            batch_embeddings.append(sentence_embedding)
        else:
            print("Warning: Empty sentence encountered.")

    # Concatenate the embeddings for all sentences in the batch
    output = torch.cat(batch_embeddings, dim=0).detach().numpy()

    print("Total batch processing time:", time.time() - start_batcher)
    return output
'''

""""
# Batcher function
def batcher(params, batch):
    start_batcher = time.time()
    sentence_embeddings = []
    
    # Process each sentence
    for sentence in batch:
        words = sentence if isinstance(sentence, list) else sentence.split()
        
        # Retrieve embeddings as a batch and avoid individual torch.tensor calls
        word_embeddings = [params['word_vec'].get(word, np.zeros(300)) for word in words]
        
        # Only convert to tensor once for the whole sentence
        if word_embeddings:
            sent_tensor = torch.tensor(word_embeddings, dtype=torch.float32).unsqueeze(0)
            
            # Process with encoder and avoid detaching within the loop
            sentence_embedding = params['encoder'](sent_tensor)
            sentence_embeddings.append(sentence_embedding)
        else:
            print("Warning: Empty sentence encountered.")

    # Concatenate all sentence embeddings
    output = torch.cat(sentence_embeddings, dim=0).detach().numpy()
    print("Total batch processing time:", time.time() - start_batcher)
    return output
"""
# Set SentEval parameters
params = {'task_path': '../data', 'usepytorch': True, 'kfold': 5}
params['classifier'] = {'nhid': 2048, 'optim': 'adam', 'batch_size': 128, 'tenacity': 5, 'epoch_size': 10}

# Define probing tasks
probing_tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift', 
                 'Tense', 'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']

print("DEBUG: Initializing SentEval engine")
# Initialize SentEval
se = senteval.engine.SE(params, batcher, prepare)
print("DEBUG: SentEval engine initialized")


# Minimal sample list to mimic real samples for prepare function
#sample_sentences = [["This", "is", "a", "sample", "sentence"], ["Another", "test", "sample"]]

# Run prepare with sample data
#prepare(params, sample_sentences)

# Define a smaller batch for testing
#small_batch = [["This", "is", "a", "test", "sentence"], ["Another", "example", "sentence"]]

# Run batcher with the smaller batch for debugging
#output = batcher(params, small_batch)
#print("Output for small batch:", output)

# Run evaluation
results = se.eval(probing_tasks)
print(results)

print("Total script execution time:", time.time() - start_script)
