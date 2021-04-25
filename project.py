from torch.utils.data.dataset import random_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os

icustays = pd.read_csv('ICUSTAYS.csv')
diagnoses = pd.read_csv('DIAGNOSES_ICD.csv')
diagnoses['ICD9_CODE'] = 'D_' + diagnoses['ICD9_CODE'].astype(str)
procedures = pd.read_csv('PROCEDURES_ICD.csv')
procedures['ICD9_CODE'] = 'P_' + procedures['ICD9_CODE'].astype(str)


icu_diag = icustays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME']].join(
    diagnoses[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE']].set_index(['HADM_ID', 'SUBJECT_ID']), on=['HADM_ID', 'SUBJECT_ID']).dropna()

icu_diag = icu_diag.groupby(['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME'])[
    'ICD9_CODE'].apply(','.join).reset_index()

icu_proc = icustays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME']].join(
    procedures[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE']].set_index(['HADM_ID', 'SUBJECT_ID']), on=['HADM_ID', 'SUBJECT_ID']).dropna()

icu_proc = icu_proc.groupby(['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME'])[
    'ICD9_CODE'].apply(','.join).reset_index()

icu_proc = icu_proc.rename(columns={'ICD9_CODE': 'ICD9_CODE_PROC'})

icu_diag = icu_diag.rename(columns={'ICD9_CODE': 'ICD9_CODE_DIAG'})

icu_diag_proc = icu_diag.join(icu_proc.set_index(['HADM_ID', 'SUBJECT_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME']), on=[
                              'HADM_ID', 'SUBJECT_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME']).dropna()

icu_diag_proc['ICD9_CODE'] = icu_diag_proc['ICD9_CODE_DIAG'] + \
    ',' + icu_diag_proc['ICD9_CODE_PROC']

icu_diag_proc = icu_diag_proc.drop(
    columns=['ICD9_CODE_DIAG', 'ICD9_CODE_PROC'])

last_icu = icu_diag_proc.sort_values(by=['SUBJECT_ID', 'INTIME']).groupby(
    'SUBJECT_ID')[['ICUSTAY_ID', 'HADM_ID', 'INTIME', 'OUTTIME']].nth(-1).reset_index()

second_last_icu = icu_diag_proc.sort_values(by=['SUBJECT_ID', 'INTIME']).groupby(
    'SUBJECT_ID')[['ICUSTAY_ID', 'HADM_ID', 'INTIME', 'OUTTIME']].nth(-2).reset_index()

last_second_last_icu = last_icu[['SUBJECT_ID', 'INTIME']].join(second_last_icu[[
                                                               'SUBJECT_ID', 'OUTTIME']].set_index(['SUBJECT_ID']), on=['SUBJECT_ID']).dropna().reset_index()

last_second_last_icu['READMISSION_WITHIN_30_DAYS'] = pd.to_datetime(
    last_second_last_icu['INTIME']) - pd.to_datetime(last_second_last_icu['OUTTIME']) < pd.Timedelta(30, 'D')

curr_numeric_code = 0
types = {}
for index, row in diagnoses.iterrows():
    curr_icd_code = row['ICD9_CODE']
    if curr_icd_code not in types:
        types[curr_icd_code] = curr_numeric_code
        curr_numeric_code += 1

for index, row in procedures.iterrows():
    curr_icd_code = row['ICD9_CODE']
    if curr_icd_code not in types:
        types[curr_icd_code] = curr_numeric_code
        curr_numeric_code += 1


icu_diag_proc = icu_diag_proc.sort_values(['SUBJECT_ID', 'INTIME'])
icd_codes_by_subject_id = {}
for index, row in icu_diag_proc.iterrows():
    curr_visit = [types[code] for code in row['ICD9_CODE'].split(',')]
    if row['SUBJECT_ID'] in icd_codes_by_subject_id:
        icd_codes_by_subject_id[row['SUBJECT_ID']].append(curr_visit)
    else:
        icd_codes_by_subject_id[row['SUBJECT_ID']] = [curr_visit]

readmitted_by_subject_id = {}
for index, row in last_second_last_icu.iterrows():
    readmitted_by_subject_id[row['SUBJECT_ID']
                             ] = row['READMISSION_WITHIN_30_DAYS']


seqs = []
pids = []
readmitted = []
for patient_id in icd_codes_by_subject_id:
    pids.append(patient_id)
    seqs.append(icd_codes_by_subject_id[patient_id])
    if patient_id in readmitted_by_subject_id and readmitted_by_subject_id[patient_id]:
        readmitted.append(True)
    else:
        readmitted.append(False)

morts = readmitted

with open('mimic3.pids', 'wb') as handle:
    pickle.dump(pids, handle)

with open('mimic3.morts', 'wb') as handle:
    pickle.dump(morts, handle)

with open('mimic3.seqs', 'wb') as handle:
    pickle.dump(seqs, handle)

with open('mimic3.types', 'wb') as handle:
    pickle.dump(types, handle)

pids = pickle.load(open('mimic3.pids', 'rb'))
morts = pickle.load(open('mimic3.morts', 'rb'))
seqs = pickle.load(open('mimic3.seqs', 'rb'))
types = pickle.load(open('mimic3.types', 'rb'))


class CustomDataset(Dataset):

    def __init__(self, seqs, morts):
        """
        TODO: Store `seqs`. to `self.x` and `morts` to `self.y`.

        Note that you DO NOT need to covert them to tensor as we will do this later.
        Do NOT permute the data.
        """

        # your code here
        self.x = seqs
        self.y = morts

    def __len__(self):
        """
        TODO: Return the number of samples (i.e. patients).
        """

        # your code here
        return len(self.y)

    def __getitem__(self, index):
        """
        TODO: Generates one sample of data.

        Note that you DO NOT need to covert them to tensor as we will do this later.
        """

        # your code here
        return [self.x[index], self.y[index]]


dataset = CustomDataset(seqs, morts)


def collate_fn(data):
    """
    TODO: Collate the the list of samples into batches. For each patient, you need to pad the diagnosis
        sequences to the sample shape (max # visits, max # diagnosis codes). The padding infomation
        is stored in `mask`.

    Arguments:
        data: a list of samples fetched from `CustomDataset`

    Outputs:
        x: a tensor of shape (# patiens, max # visits, max # diagnosis codes) of type torch.long
        masks: a tensor of shape (# patiens, max # visits, max # diagnosis codes) of type torch.bool
        rev_x: same as x but in reversed time. This will be used in our RNN model for masking 
        rev_masks: same as mask but in reversed time. This will be used in our RNN model for masking
        y: a tensor of shape (# patiens) of type torch.float

    Note that you can obtains the list of diagnosis codes and the list of mortality labels
        using: `sequences, labels = zip(*data)`
    """

    sequences, labels = zip(*data)

    # your code here
    max_visits = max([len(person) for person in sequences])
    max_codes = max([len(item) for sequence in sequences for item in sequence])
    x = []
    masks = []
    rev_x = []
    rev_masks = []
    y = []
    for i, s in enumerate(sequences):
        l = labels[i]
        x += [[visit + [0] * (max_codes - len(visit))
               for visit in s] + [[0] * max_codes] * (max_visits - len(s))]
        masks += [[[1] * len(visit) + [0] * (max_codes - len(visit))
                   for visit in s] + [[0] * max_codes] * (max_visits - len(s))]
        rev_x += [list(reversed([visit + [0] * (max_codes - len(visit))
                                 for visit in s])) + [[0] * max_codes] * (max_visits - len(s))]
        rev_masks += [list(reversed([[1] * len(visit) + [0] * (max_codes - len(visit))
                                     for visit in s])) + [[0] * max_codes] * (max_visits - len(s))]
        y += [l]

    x = torch.LongTensor(x)
    masks = torch.BoolTensor(masks)
    rev_x = torch.LongTensor(rev_x)
    rev_masks = torch.BoolTensor(rev_masks)
    y = torch.FloatTensor(y)
    return x, masks, rev_x, rev_masks, y


split = int(len(dataset)*0.8)

lengths = [split, len(dataset) - split]
train_dataset, val_dataset = random_split(dataset, lengths)

print("Length of train dataset:", len(train_dataset))
print("Length of val dataset:", len(val_dataset))


def load_data(train_dataset, val_dataset, collate_fn):
    '''
    TODO: Implement this function to return the data loader for  train and validation dataset. 
    Set batchsize to 32. Set `shuffle=True` only for train dataloader.

    Arguments:
        train dataset: train dataset of type `CustomDataset`
        val dataset: validation dataset of type `CustomDataset`
        collate_fn: collate function

    Outputs:
        train_loader, val_loader: train and validation dataloaders

    Note that you need to pass the collate function to the data loader `collate_fn()`.
    '''

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

    return train_loader, val_loader


train_loader, val_loader = load_data(train_dataset, val_dataset, collate_fn)


def sum_embeddings_with_mask(x, masks):
    """
    TODO: mask select the embeddings for true visits (not padding visits) and then
        sum the embeddings for each visit up.

    Arguments:
        x: the embeddings of diagnosis sequence of shape (batch_size, # visits, # diagnosis codes, embedding_dim)
        masks: the padding masks of shape (batch_size, # visits, # diagnosis codes)

    Outputs:
        sum_embeddings: the sum of embeddings of shape (batch_size, # visits, embedding_dim)

    NOTE: Do NOT use for loop.

    """

    # your code here
    batch_size, visits, embedding_dim = masks.shape
    return torch.sum(x * masks.reshape(batch_size, visits, embedding_dim, 1), 2)


def get_last_visit(hidden_states, masks):
    """
    TODO: obtain the hidden state for the last true visit (not padding visits)

    Arguments:
        hidden_states: the hidden states of each visit of shape (batch_size, # visits, embedding_dim)
        masks: the padding masks of shape (batch_size, # visits, # diagnosis codes)

    Outputs:
        last_hidden_state: the hidden state for the last true visit of shape (batch_size, embedding_dim)

    NOTE: DO NOT use for loop.

    HINT: Consider using `torch.gather()`.
    """

    # your code here
    sum_masks = masks.sum(axis=2)
    last_true_visits = ((sum_masks > 0).sum(axis=1) - 1)
    last_true_visits = last_true_visits.view(-1,
                                             1, 1).expand(hidden_states.size())
    true_h_n = torch.gather(hidden_states, dim=1,
                            index=last_true_visits)[:, -1, :]
    return true_h_n


class NaiveRNN(nn.Module):

    """
    TODO: implement the naive RNN model above.
    """

    def __init__(self, num_codes):
        super().__init__()
        """
        TODO: 
            1. Define the embedding layer using `nn.Embedding`. Set `embDimSize` to 128.
            2. Define the RNN using `nn.GRU()`; Set `hidden_size` to 128. Set `batch_first` to True.
            3. Define the linear layers using `nn.Linear()`; Set `output_size` to 1.
            4. Define the final activation layer using `nn.Sigmoid().

        Arguments:
            num_codes: total number of diagnosis codes
        """

        # your code here
        self.embeddingLayer = nn.Embedding(
            num_embeddings=num_codes, embedding_dim=128)
        self.RNN = nn.GRU(input_size=128, hidden_size=128, batch_first=True)
        self.linear = nn.Linear(in_features=128, out_features=1)
        self.finalActivationLayer = nn.Sigmoid()

    def forward(self, x, masks, rev_x, rev_masks):
        """
        TODO:
            1. Pass the sequence through the embedding layer;
            2. Sum the embeddings for each diagnosis code up for a visit of a patient.
               Use `sum_embeddings_with_mask()`;
            3. Pass the embegginds through the RNN layer;
            4. Obtain the hidden state at the last visit.
               Use `get_last_visit()`;
            5. Pass the hidden state through the linear and activation layers.

        Arguments:
            x: the diagnosis sequence of shape (batch_size, # visits, # diagnosis codes)
            masks: the padding masks of shape (batch_size, # visits, # diagnosis codes)

        Outputs:
            probs: probabilities of shape (batch_size)

        Note that rev_x, rev_masks are passed in as arguments so that we can use the same 
        training and validation function for both models. You can ignore the them here.
        """

        # your code here
        x = self.embeddingLayer(x)
        out = sum_embeddings_with_mask(x, masks)
        hidden_states = self.RNN(out)
        x = get_last_visit(hidden_states[0], masks)
        x = self.linear(x)
        probs = self.finalActivationLayer(x)
        return probs.squeeze()


# load the model here
naive_rnn = NaiveRNN(num_codes=len(types))
naive_rnn


criterion = nn.BCELoss()
optimizer = torch.optim.Adam(naive_rnn.parameters(), lr=0.001)


def eval_model(model, val_loader):
    """
    TODO: evaluate the model.

    Arguments:
        model: the RNN model
        val_loader: validation dataloader

    Outputs:
        precision: overall precision score
        recall: overall recall score
        f1: overall f1 score
        roc_auc: overall roc_auc score

    Note that please pass all four arguments to the model so that we can use this function for both 
    models. (Use `model(x, masks, rev_x, rev_masks)`.)

    HINT: checkout https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    """

    # your code here
    model.eval()
    Y_pred = np.array([])
    Y_test = np.array([])
    for x, masks, rev_x, rev_masks, target in val_loader:
        # your code here
        y_pred_tag = model(x, masks, rev_x, rev_masks)
#         _, y_pred_tag = torch.max(outputs, dim = 1)
        if Y_pred.shape == (0,):
            Y_pred = y_pred_tag.detach().numpy()
        else:
            Y_pred = np.concatenate(
                [Y_pred, y_pred_tag.detach().numpy()], axis=0)
        if Y_test.shape == (0,):
            Y_test = target.detach().numpy()
        else:
            Y_test = np.concatenate([Y_test, target.detach().numpy()], axis=0)
    precision, recall, f1, _ = precision_recall_fscore_support(
        Y_test, Y_pred > 0.5, average='binary')
    roc_auc = roc_auc_score(Y_test, Y_pred)
    return precision, recall, f1, roc_auc


def train(model, train_loader, val_loader, n_epochs):
    """
    TODO: train the model.

    Arguments:
        model: the RNN model
        train_loader: training dataloder
        val_loader: validation dataloader
        n_epochs: total number of epochs

    You need to call `eval_model()` at the end of each training epoch to see how well the model performs 
    on validation data.

    Note that please pass all four arguments to the model so that we can use this function for both 
    models. (Use `model(x, masks, rev_x, rev_masks)`.)
    """

    for epoch in range(n_epochs):
        curr_epoch_loss = []
        for x, masks, rev_x, rev_masks, target in train_loader:
            # your code here
            optimizer.zero_grad()
            outputs = model(x, masks, rev_x, rev_masks)
            loss = criterion(outputs, target)
            curr_epoch_loss.append(loss.cpu().data.numpy())
            loss.backward()
            optimizer.step()
        p, r, f, roc_auc = eval_model(model, val_loader)
        print(p, r, f, roc_auc)
        print(f"Epoch {epoch}: curr_epoch_loss={np.mean(curr_epoch_loss)}")
    return model


# number of epochs to train the model
n_epochs = 5
train(naive_rnn, train_loader, val_loader, n_epochs)

p, r, f, roc_auc = eval_model(naive_rnn, val_loader)
print(roc_auc)

def collate_fn(data):
    """
    Collate the the list of samples into batches. For each patient, you need to pad the diagnosis
    sequences to the sample shape (max # visits, max # diagnosis codes). The padding infomation
    is stored in `mask`.
    
    Arguments:
        data: a list of samples fetched from `CustomDataset`
        
    Outputs:
        x: a tensor of shape (# patiens, max # visits, max # diagnosis codes) of type torch.long
        masks: a tensor of shape (# patiens, max # visits, max # diagnosis codes) of type torch.bool
        rev_x: same as x but in reversed time.
        rev_masks: same as mask but in reversed time.
        y: a tensor of shape (# patiens) of type torch.float
        
    Note that you can obtains the list of diagnosis codes and the list of mortality labels
        using: `sequences, labels = zip(*data)`
    """

    sequences, labels = zip(*data)

    y = torch.tensor(labels, dtype=torch.float)
    
    num_patients = len(sequences)
    num_visits = [len(patient) for patient in sequences]
    num_codes = [len(visit) for patient in sequences for visit in patient]

    max_num_visits = max(num_visits)
    max_num_codes = max(num_codes)
    
    x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.long)
    rev_x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.long)
    masks = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.bool)
    rev_masks = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.bool)
    for i_patient, patient in enumerate(sequences):
        for j_visit, visit in enumerate(patient):
            l = len(visit)
            x[i_patient, j_visit, :l] = torch.tensor(visit, dtype=torch.long)
            masks[i_patient, j_visit, :l].fill_(1)
            """
            TODO: update rev_x and rev_masks. 
            """
            # your code here
            m = len(patient)
            rev_x[i_patient, m - j_visit - 1, :l] = torch.tensor(visit, dtype=torch.long)
            rev_masks[i_patient, m - j_visit - 1, :l].fill_(1)
    
    return x, masks, rev_x, rev_masks, y


class AlphaAttention(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        """
        Define the linear layer `self.a_att` for alpha-attention using `nn.Linear()`;
        
        Arguments:
            hidden_dim: the hidden dimension
        """
        
        self.a_att = nn.Linear(hidden_dim, 1)

    def forward(self, g):
        """
        TODO: Implement the alpha attention.
        
        Arguments:
            g: the output tensor from RNN-alpha of shape (batch_size, seq_length, hidden_dim) 
        
        Outputs:
            alpha: the corresponding attention weights of shape (batch_size, seq_length, 1)
            
        HINT: consider `torch.softmax`
        """
        
        # your code here
        alpha = self.a_att(g)
        alpha = torch.softmax(alpha,1)
        return alpha

class BetaAttention(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        """
        Define the linear layer `self.b_att` for beta-attention using `nn.Linear()`;
        
        Arguments:
            hidden_dim: the hidden dimension
        """
        
        self.b_att = nn.Linear(hidden_dim, hidden_dim)


    def forward(self, h):
        """
        TODO: Implement the beta attention.
        
        Arguments:
            h: the output tensor from RNN-beta of shape (batch_size, seq_length, hidden_dim) 
        
        Outputs:
            beta: the corresponding attention weights of shape (batch_size, seq_length, hidden_dim)
            
        HINT: consider `torch.tanh`
        """
        
        # your code here
        beta = self.b_att(h)
        beta = torch.tanh(beta)
        return beta

def attention_sum(alpha, beta, rev_v, rev_masks):
    """
    TODO: mask select the hidden states for true visits (not padding visits) and then
        sum the them up.

    Arguments:
        alpha: the alpha attention weights of shape (batch_size, seq_length, 1)
        beta: the beta attention weights of shape (batch_size, seq_length, hidden_dim)
        rev_v: the visit embeddings in reversed time of shape (batch_size, # visits, embedding_dim)
        rev_masks: the padding masks in reversed time of shape (# visits, batch_size, # diagnosis codes)

    Outputs:
        c: the context vector of shape (batch_size, hidden_dim)
        
    NOTE: Do NOT use for loop.
    """

    the_mask = (torch.sum(rev_masks, -1) > 0).type(torch.float).unsqueeze(-1)
    true_visit = torch.sum(alpha*beta*rev_v*the_mask, dim=1)
    return true_visit

def sum_embeddings_with_mask(x, masks):
    """
    Mask select the embeddings for true visits (not padding visits) and then sum the embeddings for each visit up.

    Arguments:
        x: the embeddings of diagnosis sequence of shape (batch_size, # visits, # diagnosis codes, embedding_dim)
        masks: the padding masks of shape (batch_size, # visits, # diagnosis codes)

    Outputs:
        sum_embeddings: the sum of embeddings of shape (batch_size, # visits, embedding_dim)
    """
    
    x = x * masks.unsqueeze(-1)
    x = torch.sum(x, dim = -2)
    return x

class RETAIN(nn.Module):
    
    def __init__(self, num_codes):
        super().__init__()
        # Define the embedding layer using `nn.Embedding`. Set `embDimSize` to 128.
        self.embedding = nn.Embedding(num_codes, 128)
        # Define the RNN-alpha using `nn.GRU()`; Set `hidden_size` to 128. Set `batch_first` to True.
        self.rnn_a = nn.GRU(128, 128, batch_first=True)
        # Define the RNN-beta using `nn.GRU()`; Set `hidden_size` to 128. Set `batch_first` to True.
        self.rnn_b = nn.GRU(128, 128, batch_first=True)
        # Define the alpha-attention using `AlphaAttention()`;
        self.att_a = AlphaAttention(128)
        # Define the beta-attention using `BetaAttention()`;
        self.att_b = BetaAttention(128)
        # Define the linear layers using `nn.Linear()`;
        self.fc = nn.Linear(128, 1)
        # Define the final activation layer using `nn.Sigmoid().
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, masks, rev_x, rev_masks):
        """
        Arguments:
            rev_x: the diagnosis sequence in reversed time of shape (# visits, batch_size, # diagnosis codes)
            rev_masks: the padding masks in reversed time of shape (# visits, batch_size, # diagnosis codes)

        Outputs:
            probs: probabilities of shape (batch_size)
        """
        # 1. Pass the reversed sequence through the embedding layer;
        rev_x = self.embedding(rev_x)
        # 2. Sum the reversed embeddings for each diagnosis code up for a visit of a patient.
        rev_x = sum_embeddings_with_mask(rev_x, rev_masks)
        # 3. Pass the reversed embeddings through the RNN-alpha and RNN-beta layer separately;
        g, _ = self.rnn_a(rev_x)
        h, _ = self.rnn_b(rev_x)
        # 4. Obtain the alpha and beta attentions using `AlphaAttention()` and `BetaAttention()`;
        alpha = self.att_a(g)
        beta = self.att_b(h)
        # 5. Sum the attention up using `attention_sum()`;
        c = attention_sum(alpha, beta, rev_x, rev_masks)
        # 6. Pass the context vector through the linear and activation layers.
        logits = self.fc(c)
        probs = self.sigmoid(logits)
        return probs.squeeze()
    

# load the model here
retain = RETAIN(num_codes = len(types))
retain

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def eval(model, val_loader):
    
    """
    Evaluate the model.
    
    Arguments:
        model: the RNN model
        val_loader: validation dataloader
        
    Outputs:
        precision: overall precision score
        recall: overall recall score
        f1: overall f1 score
        roc_auc: overall roc_auc score
        
    REFERENCE: checkout https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    """
    
    model.eval()
    y_pred = torch.LongTensor()
    y_score = torch.Tensor()
    y_true = torch.LongTensor()
    model.eval()
    for x, masks, rev_x, rev_masks, y in val_loader:
        y_logit = model(x, masks, rev_x, rev_masks)
        """
        TODO: obtain the predicted class (0, 1) by comparing y_logit against 0.5, 
              assign the predicted class to y_hat.
        """
        y_hat = None
        # your code here
        y_hat = y_logit > 0.5
        y_score = torch.cat((y_score,  y_logit.detach().to('cpu')), dim=0)
        y_pred = torch.cat((y_pred,  y_hat.detach().to('cpu')), dim=0)
        y_true = torch.cat((y_true, y.detach().to('cpu')), dim=0)
    
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    roc_auc = roc_auc_score(y_true, y_score)
    return p, r, f, roc_auc

def train(model, train_loader, val_loader, n_epochs):
    """
    Train the model.
    
    Arguments:
        model: the RNN model
        train_loader: training dataloder
        val_loader: validation dataloader
        n_epochs: total number of epochs
    """
    
    model.train()
    for epoch in range(n_epochs):
        train_loss = 0
        for x, masks, rev_x, rev_masks, y in train_loader:
            optimizer.zero_grad()
            """ 
            TODO: calculate the loss using `criterion`, save the output to loss.
            """
            loss = None
            # your code here
            outputs = model(x, masks, rev_x, rev_masks)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch+1, train_loss))
        p, r, f, roc_auc = eval(model, val_loader)
        print('Epoch: {} \t Validation p: {:.2f}, r:{:.2f}, f: {:.2f}, roc_auc: {:.2f}'
              .format(epoch+1, p, r, f, roc_auc))


# load the model
retain = RETAIN(num_codes = len(types))

# load the loss function
criterion = nn.BCELoss()
# load the optimizer
optimizer = torch.optim.Adam(retain.parameters(), lr=1e-3)

n_epochs = 5
train(retain, train_loader, val_loader, n_epochs)

p, r, f, roc_auc = eval(retain, val_loader)
print(roc_auc)