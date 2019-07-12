import numpy as np
import os
import torch
import torch.nn as nn
import pickle
import random
import time
import itertools
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import optim
from torch.utils.data import DataLoader

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

        inner_dim = 640
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
                

class MultiLayeredGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, rnn_type="gru", dropout_p=.0):
        super(MultiLayeredGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.num_layers = num_layers

        self.layer0 = getattr(nn, 'GRU')(self.input_size, self.hidden_size)
        for l in range(1, num_layers):
            setattr(self, 'layer{}'.format(l), 
                    getattr(nn, 'GRU')(self.hidden_size, self.hidden_size))

    def _forward(self, input, hidden):
        output, hidden = self.layer0(input, hidden)
        return output, hidden
        
    def forward(self, input, hidden):
        # input.shape (T, B, H)
        # hidden.shape (n_layers, B, H)
        hidden = [h.unsqueeze(0) for h in hidden]
        output, hidden[0] = self.layer0(input, hidden[0])
        # output.shape (T, B, H)
        # hidden[0].shape (1,B,H)
        for l in range(1, self.num_layers):
            output, hidden[l] = getattr(self, 'layer{}'.format(l))(output, hidden[l])
        hidden = [h.squeeze(0) for h in hidden]
        return output, hidden


class AttentionDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, 
            dropout_p=.0, tf_prob=.0, att_type='content', num_layers=3):
        super(AttentionDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.teacher_forcing_prob = tf_prob
        self.max_input_len = MAX_INPUT_LEN
        self.att_type = att_type
        self.num_layers = num_layers

        self.att_hdim = 64

        self.emb = nn.Embedding(self.output_size, self.hidden_size)
    

        if att_type == 'content':
            self.attn_f1 = nn.Tanh()
            self.attn_f2 = nn.Tanh()
#            self.attn_f1 = nn.Sequential(
#                    nn.Linear(self.hidden_size, self.hidden_size, bias=False),
#                    nn.ReLU(),
#                    nn.Linear(self.hidden_size, self.att_hdim, bias=False))
#
#            self.attn_f2 = nn.Sequential(
#                    nn.Linear(self.hidden_size, self.hidden_size, bias=False),
#                    nn.ReLU(),
#                    nn.Linear(self.hidden_size, self.att_hdim, bias=False))
        elif att_type == 'location':
            self.attn_loc = nn.Linear(self.hidden_size*2, self.max_input_len)
    
        #self.gru = nn.GRU(self.hidden_size*2, self.hidden_size)
        self.gru = MultiLayeredGRU(self.hidden_size*2, self.hidden_size, num_layers=num_layers)

        self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size*2, self.output_size))
#                nn.ReLU(),
#                nn.Linear(self.hidden_size, self.output_size))


    def forward(self, y_inputs, y_outputs, y_enc):
        # y_inputs: (T, B)
        # h: (num_layers, B, Hdim)
        # y_enc: (T, B, Hdim)

        y_input_t = y_inputs[0]
#        hidden = y_enc.new_zeros([self.gru.num_layers, y_inputs.shape[1], self.hidden_size])
        hidden = [y_enc.new_zeros([y_inputs.shape[1], self.hidden_size])] * self.num_layers
        context_t = y_enc.new_zeros([y_inputs.shape[1], self.hidden_size])
        
        max_time_step = y_inputs.shape[0]
        tf = True if random.random() < self.teacher_forcing_prob else False
    
        outputs = []; attentions = []
        for t in range(max_time_step):
            output_t, hidden, attn_t, context_t = \
                    self._forward_t(y_input_t, hidden, y_enc, context_t)

            outputs.append(output_t)
            attentions.append(attn_t)

            if tf:
                if not t==max_time_step-1:
                    y_input_t = y_inputs[t+1]
            else:
                top_v, top_i = output_t.topk(1)
                predicted = top_i.squeeze(dim=-1).detach()
                y_input_t = predicted
        
        predicted_seq = torch.stack(outputs, dim=0)
        attention_seq = torch.stack(attentions, dim=1)
        pdb.set_trace()
        
        ce_loss = F.cross_entropy(predicted_seq.view(-1, self.output_size), 
                y_outputs.view(-1), ignore_index=PAD_TOKEN, reduction='mean')
        

        a = ''.join([libri.decode_mapping[p.item()] \
                for p in predicted_seq[:,0].argmax(1) if p.item() is not PAD_TOKEN])
        b = ''.join([libri.decode_mapping[p.item()] \
                for p in y_outputs[:,0] if p.item() is not PAD_TOKEN])
    

#        print (a)
#        print (b)
#        print ('-'*20)

        return predicted_seq, attention_seq, ce_loss

    
    def _forward_t(self, input, hidden, encoder_outputs, context):
        # input.shape : (n_batches)_t
        # hidden.shape : (num_layers, n_batches, hidden_dim)_t
        # encoder_o.shape : (iT, n_batches, h_idm)
        # context.shape (n_batches, h_dim)
        input = self.emb(input)
        rnn_input = torch.cat((input, context), 1).unsqueeze(0) # add time dim
        output, hidden = self.gru(rnn_input, hidden)
        
        if self.att_type=='content':
            attn_w, context = self.attention(output.squeeze(dim=0), encoder_outputs, i)
        elif self.att_type=='location':
            attn_w, context = self.loc_attention(input, hidden.squeeze(dim=0), encoder_outputs)


        output = torch.cat((output.squeeze(dim=0), context), 1)
        output = self.classifier(output)
        # output: (B,O)
        # hidden: (NL, B, H)
        # attn_w: (B, iT)
        # context: (B,H)
        return output, hidden, attn_w, context

    def loc_attention(self, input, h_s, encoder_outputs):
        # h_s.shape: (batch, hdim)
        # input.shape : (batch, h_dim)
        # encoder_o.shape; (T,batch,hdim)
        input = torch.cat((input, h_s), 1)
        attw = F.softmax(self.attn_loc(input), dim=1)
        # attw.shape: (b,t)

        eh = encoder_outputs.transpose(1,0)
        # eh (B,T,hdim)
        context = torch.bmm(attw.unsqueeze(1), eh).squeeze(1)
        return attw, context

    def attention(self, h_s, encoder_outputs, i):
        # h_s.shape: (batches, h_dim)
        # encoder_outputs.shape (T, batches, h_dim)
        
        context_1 = self.attn_f1(h_s)
        context_1 = context_1.unsqueeze(1)
        # context_1.shape: (batch, 1, attdim)

        context_2 = self.attn_f2(encoder_outputs)
        context_2 = context_2.permute(1,2,0)
        # context_2.shape: (batch, atthdim, T)

        attw = torch.bmm(context_1, context_2).squeeze(dim=1) * 1.
        attw = F.softmax(attw, dim=1)
        # attw.shape: (batch, T)

        eh = encoder_outputs.transpose(1,0)
        # eh.shape: (batch, T, hdim)
        context = torch.bmm(attw.unsqueeze(1), eh).squeeze(dim=1)
        
        return attw, context


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__() 
        self.n_layers = 1
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, 
                self.n_layers, dropout=0, bidirectional=True)

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

def max_pad_seq(seq, max_len):
    if len(seq) >= max_len:
        return seq[:max_len]
    else:
        pad = seq.new_zeros([max_len-len(seq), seq.shape[1], seq.shape[2]])
        return torch.cat((seq, pad), dim=0)


def collate_fn(batch):
    # data, label
    xbs = [torch.tensor(b[0]) for b in batch] # t,d
    xlen = torch.tensor([get_padded_len(_x, n_conv_layers) for _x in xbs])

    x_batches = pad_sequence(xbs)
    ybs = [b[1][:-1] for b in batch]
    y_inputs = pad_sequence(ybs, padding_value=EOS_TOKEN).long()
    ybs = [b[1][1:] for b in batch]
    y_outputs = pad_sequence(ybs, padding_value=PAD_TOKEN).long()
    return x_batches, xlen, y_inputs, y_outputs

#def get_sorted_random_batch(dataset):
#    rnd_idx = np.random.randint(len(dataset), size=batch_size)
#    batches = [dataset[r] for r in rnd_idx]
#    xlen = [get_padded_len(_x, n_conv_layers) for _x in xbs]
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
    x_dim = 80
    c_dim = 640
    h_dim = 128
    n_conv_layers = 2
    max_epoch = 80
    learning_rate = 1e-1
    teacher_forcing_prob = 0.0
    grad_clip = 10.
    num_workers = 10
    n_decoder_layer = 1
    MAX_INPUT_LEN = get_padded_len([0 for _ in range(2000)], n_conv_layers)
    pretrained = False
    fdata_loc = 'data/libri_fbank80_char30'
    save_loc = 'bs{}.hdim{}.cdim{}.convn{}.lr{}.tfp{}'\
            .format(batch_size, h_dim, c_dim, n_conv_layers,
                    learning_rate, teacher_forcing_prob)
    
    device = 4
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(device)

    libri = LibiriSpeech(fdata_loc, 'train')

    data_loader = DataLoader(libri, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, 
            drop_last=True, collate_fn=collate_fn)
    EOS_TOKEN = libri.encode_mapping['<eos>']
    SOS_TOKEN = libri.encode_mapping['<sos>']
    PAD_TOKEN = libri.encode_mapping['<pad>']

    max_iter = len(libri) // batch_size
    np.random.seed(2019)
    # ------------- model construction ------------
    convnet = EmbeddingNet(x_dim, c_dim, n_conv_layers)
    convnet.cuda()

    encoder = EncoderRNN(c_dim, h_dim)
    encoder.cuda()

    decoder = AttentionDecoderRNN(h_dim, libri.n_classes, 
            tf_prob=teacher_forcing_prob, att_type='content', num_layers=n_decoder_layer) 
    decoder.cuda()

    if pretrained:
        _save_loc = 'models/{}_' + save_loc
        encoder.load_state_dict(torch.load(_save_loc.format('encoder')))
        decoder.load_state_dict(torch.load(_save_loc.format('decoder')))
        convnet.load_state_dict(torch.load(_save_loc.format('convnet')))
        print ('loaded from {}'.format(_save_loc))

    
    full_param = [convnet.parameters(), encoder.parameters(), decoder.parameters()]
    optimizer = optim.SGD(itertools.chain(*full_param), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
            [30,50], gamma=0.1)
    
    # --------------- training ----------------
    stats_loss = 0.
    t0 = time.time()
    pt0 = time.time()
    accm = [.0, .0, .0, .0, .0]
    for epoch in range(max_epoch):
        
        for i, batches in enumerate(data_loader):


            named = ["x_batches", "xlen", "y_inputs", "y_outputs"]
            for cc, elem in enumerate(batches):
                exec("{}=batches[{}].cuda()".format(named[cc],cc))

            pt1 = time.time()


            optimizer.zero_grad()
            # model forward path
            x_batches = convnet(x_batches)

            pt2 = time.time()

            out_enc, _ = encoder(x_batches, xlen)

            pt3 = time.time()
            #out_enc = max_pad_seq(out_enc, MAX_INPUT_LEN)
            prediction, attention, ce_loss = decoder(y_inputs, y_outputs, out_enc)

            pt4 = time.time()
            
            stats_loss += ce_loss.item()
            ce_loss.backward()
            
            _ = nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
            _ = nn.utils.clip_grad_norm_(convnet.parameters(), grad_clip)
            _ = nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

            
            optimizer.step()
            
    #        for name, param in encoder.named_parameters():
    #            print("Mean of Grad of {}: {:.2e}".format(name, torch.mean(torch.abs(param.grad))))
    #            print (torch.sqrt(torch.sum(param.grad**2)))
    #        for name, param in decoder.named_parameters():
    #            print("Mean of Grad of {}: {:.2e}".format(name, torch.mean(torch.abs(param.grad))))
    #            print (torch.sqrt(torch.sum(param.grad**2)))
            
            accm[0] += pt1-pt0
            accm[1] += pt2-pt1
            accm[2] += pt3-pt2
            accm[3] += pt4-pt3
            pt0 = time.time()
            accm[4] += pt0-pt4

            if i % 50 == 0 and i != 0:
                print ('epoch {:2d} ( {:5d} / {:5d} )  | loss {:.3f} in {:.3f} sec'\
                        .format(epoch, i, max_iter, stats_loss / i, time.time()-t0))
                _save_loc = 'models/{}_' + save_loc
                np.save(_save_loc.format('att'), attention.data.cpu().numpy())

                print ('profiling:::')
                print ('batching time: {:.3f} \n'
                        'convnet: {:.3f} \n'
                        'encdoer: {:.3f} \n'
                        'decoder: {:.3f} \n'
                        'update:  {:.3f}'.format(accm[0],
                            accm[1], accm[2], accm[3], accm[4]))
                
                accm = [.0, .0, .0, .0, .0]



        print ('=======================EPOCH {}========================='.format(epoch))
        print ('loss : {:.3f} in {:.3f} secs'\
                .format(stats_loss / max_iter, time.time()-t0))
        stats_loss = .0
        t0 = time.time()
        lr_scheduler.step()
            
        if epoch % 5 == 0 and i != 0: 
            _save_loc = 'models/{}_' + save_loc
            torch.save(encoder.state_dict(), _save_loc.format('convnet'))
            torch.save(encoder.state_dict(), _save_loc.format('encoder'))
            torch.save(convnet.state_dict(), _save_loc.format('decoder'))
            print ('model saved as {}'.format(_save_loc))




           
           
























    #
