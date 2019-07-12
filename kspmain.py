import numpy as np
import os
import torch
import torch.nn as nn
import pickle
import random
import time
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import optim
from pcm_reader import Kspeech

import pdb


class LibiriSpeech():
    def __init__(self, data_dir, phase='train'):
        self.data_dir = data_dir

        with open(os.path.join(data_dir, 'mapping.pkl'), 'rb') as f:
            self.encode_mapping = pickle.load(f)
            self.encode_mapping['<pad>'] = len(self.encode_mapping.keys())
        self.decode_mapping = {y:x for x,y in self.encode_mapping.items()}
        
        self.n_classes = len(self.encode_mapping)
        trans_str = {'train': 'train-clean-100', 'test': 'test-clean', 
                'dev': 'dev-clean', 'val': 'dev-clean'}
        self.phase = trans_str[phase]

        filenames = os.listdir(os.path.join(data_dir, self.phase))
        with open(os.path.join(data_dir, self.phase + '.csv')) as f:
            labels = f.readlines()
        
        _y = [lb.split(',')[2].split('_') \
                for lb in labels[1:]]
        self.y = []
        for y in _y:
            # vectorize or not ?
            self.y.append(torch.Tensor(\
                [int(y_i) for y_i in y]))

        self.x = [os.path.join(data_dir, lb.split(',')[1]) for lb in labels[1:]]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        fname = self.x[idx]
        data = np.load(fname) # (T, F)
        label = self.y[idx]
        return data, label


class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super(Block, self).__init__()
        self.conv1 = nn.Conv1d(n_in, n_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(n_out)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class EmbeddingNet(nn.Module):
    def __init__(self, dim_in, dim_out, n_conv_layers):
        super(EmbeddingNet, self).__init__()

        inner_dim = 64
        layers = [Block(dim_in, inner_dim)]
        for i in range(n_conv_layers-2):
            layers.append(Block(inner_dim, inner_dim))
        layers.append(Block(inner_dim, dim_out))
        self.backbone = nn.Sequential(*layers)

    def forward(self, x):
        # x.shape: (T,B,D)
        x = x.permute(1,2,0)
        # x.shape: (B,D,T)

        x = self.backbone(x)
        # x.shape (B,256,T//8?)
        x = x.permute(2,0,1)
        return x
                

class AttentionDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=.0, tf_prob=.0):
        super(AttentionDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.teacher_forcing_prob = tf_prob

        self.att_hdim = 64

        self.emb = nn.Embedding(self.output_size, self.hidden_size)

        self.attn_f1 = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.att_hdim))

        self.attn_f2 = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.att_hdim))
    
        self.gru = nn.GRU(self.hidden_size*2, self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.gru2 = nn.GRU(self.hidden_size, self.hidden_size)


    def forward(self, y_inputs, y_outputs, y_enc, h_enc, i):
        # y_inputs: (T, B)
        # h_enc: (B, Hdim)
        # y_enc: (T, B, Hdim)

        y_input_t = y_inputs[0]
        h_enc = h_enc.new_zeros(h_enc.shape)
        context_t = h_enc.new_zeros(h_enc.shape)
        max_time_step = y_inputs.shape[0]
        tf = True if random.random() < self.teacher_forcing_prob else False
    
        outputs = []; attentions = []
        for t in range(max_time_step):
            y_dec_t, h_dec_t, attw_t, context_t = self._forward_t(
                    y_input_t, h_enc, y_enc, context_t, i)
            outputs.append(y_dec_t.squeeze())
            attentions.append(attw_t)

            h_enc = h_dec_t.squeeze()
            if tf:
                if not t==max_time_step-1:
                    y_input_t = y_inputs[t+1]
            else:
                top_v, top_i = y_dec_t.squeeze().topk(1)
                predicted = top_i.squeeze().detach()
                y_input_t = predicted
        

        
        predicted_seq = torch.stack(outputs, dim=0)
        attention_seq = torch.stack(attentions, dim=1)
        
        ce_loss = F.cross_entropy(predicted_seq.view(-1, self.output_size), 
                y_outputs.view(-1), ignore_index=PAD_TOKEN, reduction='mean')
        

#        a = ''.join([libri.decode_mapping[p.item()] \
#                for p in predicted_seq[:,0].argmax(1) if p.item() is not PAD_TOKEN])
#        b = ''.join([libri.decode_mapping[p.item()] \
#                for p in y_outputs[:,0] if p.item() is not PAD_TOKEN])
        pdb.set_trace()
        if i % 500 == 0 and i != 0:
            np.save('temp.npy', attention_seq.data.cpu().numpy())

#        for v in self.named_parameters():
#            print (v)
#            pdb.set_trace()

#        print (a)
#        print (b)
#        print ('-'*20)

        return predicted_seq, ce_loss

    
    def _forward_t(self, input, hidden, encoder_outputs, context, i):
        # input.shape : (n_batches)_t
        # hidden.shape : (n_batches, hidden_dim)_t
        # encoder_o.shape : (T, n_batches, h_idm)
        # context.shape (n_batches, h_dim)
        input = self.emb(input)
        rnn_input = torch.cat((input, context), 1).unsqueeze(0)
        _, hidden = self.gru(rnn_input, hidden.unsqueeze(0))

#        output = self.fc1(output.squeeze())
#        output = F.relu(output)
#        output = self.fc2(output)
#       
        attn_w, context = self.attention(hidden.squeeze(), encoder_outputs, i)
        
        output = torch.cat((hidden.squeeze(), context), 1)
        output = self.fc1(output)
        output = F.relu(output)
        output = self.fc2(output)
            
        return output, hidden, attn_w, context



    def attention(self, h_s, encoder_outputs, i):
        # h_s.shape: (batches, h_dim)
        # encoder_outputs.shape (T, batches, h_dim)
        
        context_1 = self.attn_f1(h_s)
        context_1 = torch.tanh(context_1)
        context_1 = context_1.unsqueeze(1)
        # context_1.shape: (batch, 1, attdim)

        context_2 = self.attn_f2(encoder_outputs)
        context_2 = torch.tanh(context_2)
        context_2 = context_2.permute(1,2,0)
        # context_2.shape: (batch, atthdim, T)
        
        attw = torch.bmm(context_1, context_2).squeeze() * 1.
        attw = F.softmax(attw, dim=1)
        # attw.shape: (batch, T)

        eh = encoder_outputs.transpose(1,0)
        # eh.shape: (batch, T, hdim)
        context = torch.bmm(attw.unsqueeze(1), eh).squeeze()
        
        return attw, context


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__() 
        self.n_layers = 1
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, 
                self.n_layers, dropout=0, bidirectional=True)

#    def forward(self, input_seq, input_length, hidden=None):
#        # input.shape : (seq_len, batch, input_size)
#        input_seq = input_seq.transpose(1,0)
#        packed = nn.utils.rnn.pack_padded_sequence(input_seq, input_length, batch_first=True, enforce_sorted=False)
#        outputs, hidden = self.gru(packed, hidden)
#        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
#        outputs = outputs.transpose(1,0)
#        # if bidirectional
#        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]
#        # ouputs.shape : (T,B,hdim)
#        # hidden.shape : (2,B,hdim)
#        hidden = hidden.sum(0)
#        return outputs, hidden

    def forward(self, input_seq, input_length, hidden=None):
        # input.shape : (seq_len, batch, input_size)

        packed = nn.utils.rnn.pack_padded_sequence(input_seq, input_length, enforce_sorted=False)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # if bidirectional
        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]
        # ouputs.shape : (T,B,hdim)
        # hidden.shape : (2,B,hdim)
        hidden = hidden.sum(0)
        return outputs, hidden

def get_padded_len(seq, n_padded):
    out = len(seq)
    for _ in range(n_padded):
        out = out // 2
    return out 

#def get_sorted_random_batch(dataset):
#    rnd_idx = np.random.randint(len(dataset), size=batch_size)
#    batches = [dataset[r] for r in rnd_idx]
#    xlen = [get_padded_len(_x, n_conv_layers) for _x in xbs]
#    pdb.set_trace()
#
#    xbs = [torch.tensor(b[0]) for b in batches] # t,d
#    xlen = torch.tensor([get_padded_len(_x, n_conv_layers) for _x in xbs]).cuda()
#    #xlen = torch.tensor([len(_x) for _x in xbs]).cuda()
#
#    x_batches = pad_sequence(xbs).cuda()
#
#
#    ybs = [b[1][:-1] for b in batches]
#    y_inputs = pad_sequence(ybs, padding_value=EOS_TOKEN).long().cuda()
#    ybs = [b[1][1:] for b in batches]
#    y_outputs = pad_sequence(ybs, padding_value=PAD_TOKEN).long().cuda()


if __name__=='__main__':

    # ------------- hyperparams --------------
    batch_size = 32
    x_dim = 129
    h_dim = 128
    c_dim = 128
    n_conv_layers = 3
    max_iter = 50000
    learning_rate = 1e-1
    teacher_forcing_prob = 1.0
    grad_clip = 20.
    pretrained = True
    #fdata_loc = 'data/libri_fbank80_char30'
    fdata_loc = '/home/mike/DataSet/speech.datasets/encoded/KsponSpeech'
    save_loc = 'bs{}.hdim{}.cdim{}.convn{}.lr{}.tfp{}'\
            .format(batch_size, h_dim, c_dim, n_conv_layers,
                    learning_rate, teacher_forcing_prob)


    device = 4
    torch.cuda.set_device(device)

    libri = Kspeech(fdata_loc, 'train')
    EOS_TOKEN = libri.EOS_token 
    SOS_TOKEN = libri.SOS_token 
    PAD_TOKEN = libri.PAD_token 
    
    np.random.seed(0)
    # ------------- model construction ------------
    encoder = EmbeddingNet(x_dim, c_dim, n_conv_layers)
    encoder.cuda()

#    encoder = EncoderRNN(c_dim, h_dim)
#    encoder.cuda()

    decoder = AttentionDecoderRNN(h_dim, libri.n_classes, tf_prob=teacher_forcing_prob) 
    decoder.cuda()

    if pretrained:
        _save_loc = 'models/{}_' + save_loc
        encoder.load_state_dict(torch.load(_save_loc.format('encoder')))
        decoder.load_state_dict(torch.load(_save_loc.format('decoder')))
        print ('loaded from {}'.format(_save_loc))

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    # --------------- training ----------------
    stats_loss = []
    t0 = time.time()
    for i in range(max_iter):

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # data batch : TODO - Loader needed
        rnd_idx = np.random.randint(len(libri), size=batch_size)
        batches = [libri[r] for r in rnd_idx]
        xbs = [torch.tensor(b[0]) for b in batches] # t,d
        xlen = torch.tensor([get_padded_len(_x, n_conv_layers) for _x in xbs]).cuda()
        #xlen = torch.tensor([len(_x) for _x in xbs]).cuda()

        x_batches = pad_sequence(xbs).cuda()
        

        ybs = [b[1][:-1] for b in batches]
        y_inputs = pad_sequence(ybs, padding_value=EOS_TOKEN).long().cuda()
        ybs = [b[1][1:] for b in batches]
        y_outputs = pad_sequence(ybs, padding_value=PAD_TOKEN).long().cuda()


        out_enc = encoder(x_batches)
        h_enc = torch.zeros([batch_size, h_dim]).cuda()
        # model forward path
#        out_enc, h_enc = encoder(x_batches, xlen)
        prediction, ce_loss = decoder(y_inputs, y_outputs, out_enc, h_enc, i)

        ce_loss.backward()
        
        _ = nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
        _ = nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

        encoder_optimizer.step()
        decoder_optimizer.step()

#        print ('iteration : {},  loss: {} '.format(i, ce_loss.item()))

        stats_loss.append(ce_loss.item())
        if i % 50 == 0:
            print ('iteration {} | loss {:.3f} in {:.2f} sec'\
                    .format(i, np.mean(stats_loss), time.time()-t0))
            stats_loss = []
            t0 = time.time()
            
        if i % 500 == 0 and i != 0: 
            _save_loc = 'models/{}_' + save_loc
            torch.save(encoder.state_dict(), _save_loc.format('encoder'))
            torch.save(decoder.state_dict(), _save_loc.format('decoder'))
            print ('model saved as {}'.format(_save_loc))




   
   
























    #
