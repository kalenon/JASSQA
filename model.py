import torch
import torch.nn as nn
    
class Ac_embed_3(nn.Module):
    def __init__(self, vocab_size_ac, n_codebook_ac, d_model=128, dropout=0.2, 
                 transformer_d_model=512, n_head=8, n_layers=2, dim_feedforward=1024):
        super().__init__()
        self.d_model = d_model
        self.n_codebook_ac = n_codebook_ac

        self.embedding_layers_ac = nn.ModuleList(
            [nn.Embedding(vocab_size_ac, d_model) for _ in range(n_codebook_ac)]
        )
        
        # 初始投影层，将多码本特征融合并投影到 Transformer 的维度
        self.multi_stream_projection_ac = nn.Sequential(
            nn.Linear(d_model * n_codebook_ac, transformer_d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True 
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, tokens_ac, lengths_ac=None):
        
        b_ac, _, t_ac = tokens_ac.size()
        mask_ac = torch.arange(t_ac, device=tokens_ac.device)[None, :] >= lengths_ac[:, None]

        src_ac = [self.embedding_layers_ac[i](tokens_ac[:, i, :]).unsqueeze(1) for i in range(self.n_codebook_ac)]
        src_ac = torch.cat(src_ac, dim=1)
        src_ac = src_ac.permute(0, 2, 1, 3).contiguous().view(b_ac, t_ac, self.d_model*self.n_codebook_ac)
        projected_ac = self.multi_stream_projection_ac(src_ac)

        transformer_output = self.transformer_encoder(
            projected_ac, 
            src_key_padding_mask=mask_ac
        )

        return transformer_output

class MyBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=True, batch_first=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=batch_first
        )
    def forward(self, x, lengths):
        sorted_lengths, sorted_idx = lengths.sort(descending=True)
        sorted_x = x[sorted_idx]
        
        packed_x = nn.utils.rnn.pack_padded_sequence(
            sorted_x, sorted_lengths.cpu(), batch_first=True
        )
        
        packed_output, (hidden, cell) = self.lstm(packed_x)
        
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )
        
        _, unsorted_idx = sorted_idx.sort()
        output = output[unsorted_idx]

        return output

class MosPredictor_mosanetplus_scoreq_crossatt_8(nn.Module):
    def __init__(self, vocab_size, vocab_dim, n_codebook, d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=512, dropout=0.2, embed_dim=1280):
        super().__init__()
        self.ac_embed = Ac_embed_3(vocab_size_ac=vocab_size, n_codebook_ac=n_codebook, dropout=dropout)

        self.dim_layer_ac = nn.Linear(512 + vocab_dim*n_codebook, 512)
        self.relu_ = nn.ReLU()
        self.sigmoid_ = nn.Sigmoid()

        self.mean_net_rnn_ac = MyBiLSTM(input_size = 512, hidden_size = 128, num_layers = 1, batch_first = True, bidirectional = True)
        self.mean_net_dnn_ac = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cross_attn_ac_se = nn.MultiheadAttention(128, num_heads=8, batch_first=True)
        self.self_attn_ac = nn.MultiheadAttention(128, num_heads=8, batch_first=True)

        self.dim_layer = nn.Linear(embed_dim, 512)
        self.mean_net_rnn_se = MyBiLSTM(input_size = 512, hidden_size = 128, num_layers = 1, batch_first = True, bidirectional = True)
        self.mean_net_dnn_se = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cross_attn_se_ac = nn.MultiheadAttention(128, num_heads=8, batch_first=True)
        self.self_attn_se = nn.MultiheadAttention(128, num_heads=8, batch_first=True)

        self.output_layer_quality = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        self.output_layer_intell = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
                
    def forward(self, tokens, latents, commit, whisper, lengths=None, lengths_se=None, is_continuous=False):
        ssl_feat_red = self.dim_layer(whisper.squeeze(1))
        ssl_feat_red_relu = self.relu_(ssl_feat_red)
        out_se = self.mean_net_rnn_se(ssl_feat_red_relu, lengths_se)
        out_se = self.mean_net_dnn_se(out_se)
        mask_se = torch.arange(out_se.size(1), device=ssl_feat_red_relu.device)[None, :] >= lengths_se[:, None]

        latents = latents.permute(0, 2, 1)
        tokenembed = self.ac_embed(tokens, lengths)
        cat_ac = torch.cat([latents, tokenembed], dim=2)
        embed_ac = self.dim_layer_ac(cat_ac)
        embed_ac_relu = self.relu_(embed_ac)
        
        out_ac = self.mean_net_rnn_ac(embed_ac_relu, lengths)
        out_ac = self.mean_net_dnn_ac(out_ac)
        mask_ac = torch.arange(out_ac.size(1), device=embed_ac_relu.device)[None, :] >= lengths[:, None]
        
        fused_ac, _ = self.cross_attn_ac_se(query=out_ac, key=out_se, value=out_se, key_padding_mask=mask_se)
        fused_se, _ = self.cross_attn_se_ac(query=out_se, key=out_ac, value=out_ac, key_padding_mask=mask_ac)

        selfac, _ = self.self_attn_ac(query=out_ac, key=out_ac, value=out_ac, key_padding_mask=mask_ac)
        selfse, _ = self.self_attn_se(query=out_se, key=out_se, value=out_se, key_padding_mask=mask_se)

        ac_enhanced = torch.cat((selfac, fused_ac), dim=2) 
        se_enhanced = torch.cat((selfse, fused_se), dim=2) 

        ac_enhanced = ac_enhanced.masked_fill(mask_ac.unsqueeze(-1), 0.0)
        ac_utt = torch.sum(ac_enhanced, dim=1)
        ac_utt = ac_utt / lengths.unsqueeze(1).float()

        se_enhanced = se_enhanced.masked_fill(mask_se.unsqueeze(-1), 0.0)
        se_utt = torch.sum(se_enhanced, dim=1)
        se_utt = se_utt / lengths_se.unsqueeze(1).float()

        concat_feat = torch.cat((ac_utt, se_utt), axis=1)      
        
        quality_utt = self.output_layer_quality(concat_feat)

        int_utt = self.output_layer_intell(concat_feat)

        pooled_ac = torch.sum(embed_ac.masked_fill(mask_ac.unsqueeze(-1), 0.0), dim=1) / lengths.unsqueeze(1).float()
        pooled_se = torch.mean(ssl_feat_red.squeeze(1), dim=1)
        return {
            "quality": quality_utt.squeeze(1),
            "intell": int_utt.squeeze(1),
            'pooled_ac': pooled_ac,
            'pooled_se': pooled_se
        }
