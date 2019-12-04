import pickle, os, sys, random, time
import numpy as np
import math
from collections import defaultdict, Counter
from io import BytesIO
import nltk 
import pandas as pd
import dill
#nltk.download('punkt') #to tokenize sentences.

def tokenize_data(sent_list):
    tok = [nltk.word_tokenize(sent) for sent in sent_list]
    tok = [item for sublist in tok for item in sublist]
    tok = [x.lower() for x in tok]
    tok = [word for word in tok if word.isalpha() or word in ['!','.','?',',']]
    tok = [word for word in tok if word != [] or '']
    return tok

def load_data():
    DATA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Load the data
    lines = open(DATA_DIR + '/data/cornell_movie-dialogs_corpus/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
    conv_lines = open(DATA_DIR + '/data/cornell_movie-dialogs_corpus/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

    # Create a dictionary to map each line's id with its text
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]

    print('creating list of sentences...')
    # Create a list of all of the conversations' lines' ids.
    convs = []
    for line in conv_lines[:-1]:
        _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
        convs.append(_line.split(','))
    
    # Sort the sentences into questions (inputs) and answers (targets)
    sentences = []
    for conv in convs:
        for i in range(len(conv)-1):
            sentences.append(id2line[conv[i]])

    print('splitting into training and testing set')
    #spliting train test split
    training_size = int(round(len(sentences)*0.8,0))
    train = sentences[:training_size]
    test = sentences[training_size:]

    test_tok = tokenize_data(test)
    train_tok = tokenize_data(train)
    return test_tok, train_tok

def train_ngram_lm(data, order=3):
    """
        Train n-gram language model
    """
    
    # pad (order-1) special tokens to the left
    # for the first token in the text
    order -= 1
    data = ['<S>'] * order + data #
    lm = defaultdict(Counter)
    
    # get ngrams for all sizes
    for k in range(1, order + 2):
      
      # loop through ngrams to get counts
      for i in range(len(data) - k + 1):
          # rolling window of ngrams
          ngrams = data[i:i + k]
          
          # split ngrams into previous word and next word
          next_word = ngrams.pop()
          
          # concatenate previous words with space
          ngrams = " ".join(ngrams)

          # add count of next word in ngram
          lm[ngrams].update([next_word])

    # convert Counter() object to dict
    lm = {key:  dict(values) for key, values in lm.items()}

    # normalize by counts
    for i in lm.keys():
      total = sum(lm[i].values(), 0.0)
      lm[i] = {k: v / total for k, v in lm[i].items()}
    return lm

class nGramsModel:
    def __init__(self, model):
        self.lm = model
        
    def tokenize_data(self, sent_list):
        sent_list = [sent_list]
        tok = [nltk.word_tokenize(sent) for sent in sent_list]
        tok = [item for sublist in tok for item in sublist]
        tok = [x.lower() for x in tok]
        tok = [word for word in tok if word.isalpha() or word in ['!','.','?',',']]
        tok = [word for word in tok if word != [] or '']
        return tok
    
    def generate_text(self, context="he is the", order=3, num_tok=10):
        
        # The goal is to generate new words following the context
        # If context has more tokens than the order of lm, 
        # generate text that follows the last (order-1) tokens of the context
        # and store it in the variable `history`
        order -= 1

        history = self.tokenize_data(context)[-order:]
        
        # `out` is the list of tokens of context
        # you need to append the generated tokens to this list
        out = context.split()
        punctuation = 0
        
        try:
            while punctuation < 2:
                # if the context word has more tokens than the order of lm
                if len(history) > order:
                    history = history[-(order - 1):]
                # for unigram models
                if order == 0:
                    history = []

                # concatenate history with space
                context = " ".join(history)
                # look up context distribution
                dist = self.lm[context]

                # find next word from distribution
                # maxmimum probable word

                #next_word = list(np.random.choice(list(dist.keys()), 1, list(dist.values())))
                next_word = [max(dist, key=dist.get)]

                # only output the first two sentences
                if next_word[0] in ['.','!','?']:
                    punctuation += 1
                
                # append next word to out
                out += next_word

                # update history
                history = history[1:]
                history += next_word
                
            return " ".join(out[3:])
        except:
            return "sorry, I coudn't understand that."

if __name__ == '__main__':
    print('loading data...')
    test_tok, train_tok = load_data()
    vocab = list(set(test_tok))
    order = 3
    print('training model...')
    model = train_ngram_lm(train_tok, order=order)
    print('packaging model...')
        
    myModel = nGramsModel(model=model)
    dill.settings['recurse'] = True
    dill.dump(myModel, open('baseline.pkl','wb'))