''' Python Libraries '''
import torch
import torch.nn as nn
#from torchsummary import summary
from einops import rearrange, repeat
from math import ceil

''' Local Libraries '''
from model.embedding_layer import DSW_embedding, GDSW_embedding
from model.encoder_layer import Encoder
from model.decoder_layer import Decoder
from model.attention_layer import FullAttention, AttentionLayer, TwoStageAttentionLayer



class Crossformer(nn.Module):
    def __init__(self, data_dim, in_len, out_len, seg_len, win_size = 4,
                factor=10, d_model=512, d_ff = 1024, n_heads=8, e_layers=3, 
                dropout=0.0, device=torch.device('cuda:0'),
                proj_layer=True, embed_type='GDSW'):
        super(Crossformer, self).__init__()
        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.seg_len = seg_len
        self.merge_win = win_size

        self.proj_layer = proj_layer
        self.embed = embed_type
        self.device = device

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * in_len / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * out_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        if self.embed == 'DSW': # original DSW
            self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        elif self.embed == 'GDSW': # geometric DSW 
            self.enc_value_embedding = GDSW_embedding(seg_len, d_model)
        # Positional embedding
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_in_len // seg_len), d_model))
        # Layer norm
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff, block_depth = 1, \
                                    dropout = dropout,in_seg_num = (self.pad_in_len // seg_len), factor = factor)
        
        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_out_len // seg_len), d_model))
        self.decoder = Decoder(seg_len, e_layers + 1, d_model, n_heads, d_ff, dropout, \
                                    out_seg_num = (self.pad_out_len // seg_len), factor = factor)
        
        ''' Add Projection Layer '''
        if proj_layer:
            self.norm = nn.LayerNorm([out_len, data_dim])
            self.proj = nn.Linear(data_dim,1)

    def forward(self, x_seq, x_geo):
        batch_size = x_seq.shape[0]
        if (self.in_len_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim = 1)

        if self.embed == 'DSW': # original DSW 
            x_seq = self.enc_value_embedding(x_seq)
        elif self.embed == 'GDSW': # geometric DSW
            x_seq = self.enc_value_embedding(x_seq, x_geo)

        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)
        
        enc_out = self.encoder(x_seq)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
        predict_y = self.decoder(dec_in, enc_out)

        ''' Add Projection '''
        if self.proj_layer:
            predict_y = self.norm(predict_y)
            predict_y = self.proj(predict_y).squeeze()

            ''' NOTE: dimension mismatch for 1 sample only, 
            no need to squeeze predict_y in crossformer forward call '''
            if predict_y.shape[0] != 1:
                predict_y = predict_y.squeeze()

            return predict_y[:, :self.out_len]
        else:
            return predict_y[:, :self.out_len, -1]