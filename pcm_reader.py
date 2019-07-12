import numpy as np
import os
import torch
import torch.nn as nn
from scipy import signal
import pdb



root_dir = '/home/mike/DataSet/speech.datasets/encoded/KsponSpeech'

class Kspeech():
    def __init__(self, data_dir, phase='train'):
        assert phase in ['train', 'test']
        self.data_dir = data_dir
        class_list_f = os.path.join(data_dir, '../chr_list.txt')
        with open(class_list_f, encoding='utf-8') as f:
            self.class_list = f.read()
        
        self.n_classes = len(self.class_list) + 3 # (SOS, EOS)

        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2

        self.frame_rate = 16000
        self.sample_width = 2
        self.channel = 1


        filenames = []
        if phase=='train':
            for i in range(4):
                ddir = os.path.join(data_dir, 'KsponSpeech_0{}'.format(i+1))
                folders = os.listdir(ddir)
                folders = [os.path.join(ddir, fd) for fd in folders \
                        if os.path.isdir(os.path.join(ddir, fd))]

                for fd in folders:
                    xs = os.listdir(fd)
                    filenames.extend(
                        [os.path.join(fd, x) for x in xs if '.pcm' in x])
        else:
            ddir = os.path.join(data_dir, 'KsponSpeech_05')
            folders = os.listdir(ddir)
            folders = [os.path.join(ddir, fd) for fd in folders \
                    if os.path.isdir(os.path.join(ddir, fd))]
            
            for fd in folders:
                xs = os.listdir(fd)
                filenames.extend(
                    [os.path.join(fd, x) for x in xs if '.pcm' in x])

        self.x = filenames
        # preprocessing needed

    def word2ind(self, word):
        ind = self.class_list.index(word) + 3
        return ind

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        fname = self.x[idx]
        pcm_data = np.array(np.memmap(fname, dtype='h', mode='r'))
        faxis, taxis, spec_data = signal.spectrogram(pcm_data, fs=16000)
        spec_data = np.transpose(spec_data)
        
        with open(os.path.join(fname.replace('.pcm', '.txt')), encoding='utf-8') as f:
            txt_data = f.read()
        txt_data = self._preprocess_label(txt_data)
        return spec_data, txt_data


    def _preprocess_label(self, txt):
        
        # 1.preprocess hangul data
        def isHangul(char):
            output1 = ord(char) >= 44032 and ord(char) <= 55203 # kor char
            output2 = char in [' ', '!', '?', '.'] # + special chars
            return output1 or output2

        def remove_paren(txt):
            # (3명)/(세 명) to 세 명
            output_txt = []
            remove_flag = 0
            for t in txt:
                if t=='(':
                    if remove_flag == 0:
                        remove_flag = 1
                    else:
                        remove_flag = 0
                if not remove_flag:
                    output_txt.append(t)

            return ''.join(output_txt)
    
        prep_txt = remove_paren(txt)
        output_txt = ''.join([t for t in prep_txt if isHangul(t)])

        # 2. vectorize data (onehot)
#        output_vector = np.zeros([len(output_txt), self.n_classes])
#        output_vector[np.arange(len(output_txt)), 
#                np.array([self.class_list.index(t) for t in output_txt])] = 1
#        labels = [self.class_list.index(t) for t in output_txt]
        labels = [self.SOS_token]
        labels.extend([self.word2ind(t) for t in output_txt])
        labels.extend([self.EOS_token])
        return torch.tensor(labels)

class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        return x
                

class EmbeddingNet(nn.Module):
    def __init__(self, fc=0):
        super(EmbeddingNet, self).__init__()
        self.num_fc = fc
        layers = [Block(3, 64)]
        for i in range(3):
            layers.append(Block(64, 64))

        self.backbone = nn.Sequential(*layers)
        self.fc = nn.Linear(64*5*5, self.num_fc)

    def forward(self, x):
        x = x.unsqueeze(3)
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__() 
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        self.gru = nn.GRU(hidden_size, hidden_size, 
                n_layers, dropout=0, bidirectional=True)

    def forward(self, input_seq, input_length, hidden=None):
#        embedded = self.embedding(input_seq)
        embedded = input_seq
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_length, enforce_sorted=False)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]
        return outputs, hidden



if __name__=='__main__':
    ksp = Kspeech(root_dir, phase='test')
    np.random.seed(0)
    
    x, y = ksp[0]
    rnd_idx = np.random.randint(100, size=5)
    batches = [ksp[r] for r in rnd_idx]
    xbs = [b[0] for b in batches]
    ybs = [b[1] for b in batches]

    x_len = [len(b) for b in xbs]
    y_len = [len(b) for b in ybs]
    pdb.set_trace()

#    x_batches = [np.pad(xb, ((0,np.max(x_len)-len(xb)), (0,0)), 'constant') for xb in xbs]
#    x_batches = torch.tensor(x_batches).transpose(0,1).cuda()

#    x_batches = torch.cat([np.pad(xb, ((0,np.max(x_len)-len(xb)),(0,0)), 'constant') for xb in xbs], dim=0).cuda()
#    y_batches = [np.pad(yb, (0,np.max(y_len)-len(yb)), 'constant') for yb in ybs]
#
#    encoder = EncoderRNN(129, None, n_layers=2)
#    encoder.cuda()
#
#    out, h = encoder(x_batches, torch.tensor(x_len).cuda())
#    pdb.set_trace()
#
#






















    #
