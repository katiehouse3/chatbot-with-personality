from sklearn.model_selection import GroupShuffleSplit
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel
import pickle
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup as get_linear_schedule_with_warmup
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import transformers
from itertools import chain
import argparse
import os
from azureml.core.run import Run
import copy
# get the Azure ML run object
run = Run.get_context()


def get_df_data(csv_file='/mounts/gpt2training/pytorch-gpt2/processed_data_final.csv'):
    df = pd.read_csv(csv_file, low_memory=False)
    # remove short or missing sentences
    df = df[pd.notnull(df['sentence_1'])]
    df = df[pd.notnull(df['sentence_2'])]
    df = df[df['sentence_1'].apply(lambda x: len(x.split()) >= 5)]
    df = df[df['sentence_2'].apply(lambda x: len(x.split()) >= 5)]

    # trim sentences to max length
    df['sentence_1'] = df['sentence_1'].apply(
        lambda x: x.split()[:20]).apply(lambda x: " ".join(x))
    df['sentence_2'] = df['sentence_2'].apply(
        lambda x: x.split()[:20]).apply(lambda x: " ".join(x))

    # remove examples missing genre
    genre_eval = df['genre'].apply(lambda x: eval(x))
    genre_lens = genre_eval.apply(lambda x: len(x))
    no_genre = genre_lens[genre_lens == 0].index
    df = df[~(df.index.isin(no_genre))]
    return df


class DialogueDataset(Dataset):
    """Movie dialogue conversation dataset."""
    # data processing functions

    def __init__(self, pkl_file):
        with open(pkl_file, 'rb') as f:
            self.X = pickle.load(f)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.X[idx]


def train(data_folder):
    checkpoint = False  # set to True if continuing to train our model, o/w false
    # set to True to chat with the unaltered GPT-2 model (at bottom of notebook)
    baseline = False
    model_file = '/gpt-2_epoch_0'

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2DoubleHeadsModel.from_pretrained('gpt2')
    csv_file = data_folder + '/processed_data_final.csv'

    genre_dict = {'comedy': '<comedy>',
                  'sport': '<sport>',
                  'biography': '<biography>',
                  'romance': '<romance>',
                  'action': '<action>',
                  'adventure': '<adventure>',
                  'drama': '<drama>',
                  'sci-fi': '<sci-fi>',
                  'family': '<family>',
                  'fantasy': '<fantasy>',
                  'musical': '<musical>',
                  'crime': '<crime>',
                  'thriller': '<thriller>',
                  'short': '<short>',
                  'western': '<western>',
                  'documentary': '<documentary>',
                  'horror': '<horror>',
                  'animation': '<animation>',
                  'film-noir': '<film-noir>',
                  'music': '<music>',
                  'war': '<war>',
                  'mystery': '<mystery>'}

    genres = genre_dict.keys()

    special_tokens = ["<speaker1>", "<speaker2>"] + \
        ["<" + genre + ">" for genre in genres]

    SPECIAL_TOKENS = {"bos_token": "<bos>", "eos_token": "<eos>",
                      "additional_special_tokens": special_tokens, "pad_token": "<pad>"}

    if not baseline:
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
        model.resize_token_embeddings(len(tokenizer))

    if not baseline:
        ngpu = 0
        for param in model.parameters():
            param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        model.lm_head = nn.Linear(model.lm_head.in_features, len(tokenizer))
        model.multiple_choice_head.summary = nn.Linear(
            model.multiple_choice_head.summary.in_features, 1, bias=True)

    # retrain final fc layer and mc layer for language modeling task
    device = torch.device("cuda:0" if (
        torch.cuda.is_available() and ngpu > 0) else "cpu")

    model = model.to(device)

    if checkpoint:
        model.load_state_dict(torch.load(model_file))

    pkl_file = data_folder + '/dialogue_data.pkl'

    dataset = DialogueDataset(pkl_file=pkl_file)
    data_size = dataset.__len__()
    batch_size = 4
    train_size = .8
    shuffle_dataset = True
    #random_seed = random.randint(1, 10000)
    random_seed = 42

    # use indexing info from dataset for splitting groups
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size,
                            random_state=random_seed)  # group stratified CV

    df = get_df_data(csv_file)
    for train_idx, val_idx in gss.split(df, df['sentence_2'], df['index']):
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(val_idx)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    # params
    lm_losses = []
    mc_losses = []
    total_losses = []

    lm_losses_val = []
    mc_losses_val = []
    total_losses_val = []

    iters = 0
    lm_coef = 2.0
    mc_coef = 1.0

    num_epochs = 3

    lr = 6.25e-5
    max_grad_norm = 1.0
    num_training_steps = (data_size // batch_size) * num_epochs
    warmup_proportion = 0.1
    num_warmup_steps = num_training_steps * .1

    grad_accum_steps = 8

    # In Transformers, optimizer and schedules are splitted and instantiated like this:
    # To reproduce BertAdam specific behavior set correct_bias=False
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps)  # PyTorch scheduler
    #scheduler = PiecewiseLinear(optimizer, "lr", [(0, lr), (num_epochs * len(train_loader), 0.0)])

    print("Starting Training Loop...")
    min_total_loss = 4000
    # For each epoch
    for epoch in range(num_epochs):
        # checkpoints
        if epoch > 0:
            torch.save(model.state_dict(),
                       "/gpt-2_epoch_{}".format(epoch))
        # For each batch in the dataloader
        for i, data in enumerate(train_loader, 0):
            model.train()

            input_ids = data[0]
            token_type_ids = data[1]
            mc_token_ids = data[2]
            lm_labels = data[3]
            mc_labels = data[4]

            output = model(input_ids, mc_token_ids=mc_token_ids, mc_labels=mc_labels,
                           token_type_ids=token_type_ids, lm_labels=lm_labels)

            lm_loss = output[0]
            mc_loss = output[1]

            total_loss = lm_loss * lm_coef + mc_loss * mc_coef / grad_accum_steps

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if i % grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss LM: %.4f\tLoss MC: %.4f\tLoss total:%.4f'
                      % (epoch, num_epochs, i, len(train_loader),
                         lm_loss.item(), mc_loss.item(), total_loss.item()))

            # Save Losses for plotting later
            lm_losses.append(lm_loss.item())
            mc_losses.append(mc_loss.item())
            total_losses.append(total_loss.item())

            curr_total_loss = total_loss.item()
            if curr_total_loss <= min_total_loss:
                min_total_loss = curr_total_loss
                best_model_wts = copy.deepcopy(model.state_dict())

            run.log('best_min_loss', np.float(min_total_loss))

            iters += 1
    return model


def main():
    # let user feed in 2 parameters, the dataset to mount or download
    # and the output of the final model
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        dest='data_folder', help='data folder mounting point')
    parser.add_argument('--output_dir', type=str, help='output directory')
    args = parser.parse_args()
    data_folder = args.data_folder  # access the datastore folder

    model = train(data_folder)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model, os.path.join(args.output_dir, 'model.pt'))


if __name__ == "__main__":
    main()
