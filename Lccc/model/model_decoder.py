import torch,os
from torch import nn
import math
from torch.nn.init import xavier_uniform_
import copy
from torch import Tensor
from typing import Optional

from torch.nn import functional as F

class resblock(nn.Module):
    '''
    module: Residual Block
    '''
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(resblock, self).__init__()
        self.left = nn.Sequential(
                nn.Conv2d(inchannel,int(outchannel/2),kernel_size = 1),
                # nn.LayerNorm(int(outchannel/2),dim=1),
                nn.BatchNorm2d(int(outchannel/2)),
                nn.ReLU(),
                nn.Conv2d(int(outchannel/2), int(outchannel / 2), kernel_size = 3, stride=1, padding=1),
                # nn.LayerNorm(int(outchannel/2),dim=1),
                nn.BatchNorm2d(int(outchannel / 2)),
                nn.ReLU(),
                nn.Conv2d(int(outchannel/2),outchannel,kernel_size = 1),
                # nn.LayerNorm(int(outchannel / 1),dim=1)
                nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x
        out = out + residual
        return F.relu(out)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        self.embedding_1D = nn.Embedding(52, int(d_model))
    def forward(self, x):
        # fixed
        x = x + self.pe[:x.size(0), :]
        # learnable
        # x = x + self.embedding_1D(torch.arange(52).cuda()).unsqueeze(1).repeat(1,x.size(1),  1)
        return self.dropout(x)

class Mesh_TransformerDecoderLayer(nn.Module):

    __constants__ = ['batch_first', 'norm_first']
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Mesh_TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(int(d_model), nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()


        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.xavier_uniform_(self.fc_alpha3.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)
        nn.init.constant_(self.fc_alpha3.bias, 0)


    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        self_att_tgt = self.norm1(tgt + self._sa_block(tgt, tgt_mask, tgt_key_padding_mask))
        # # cross self-attention
        enc_att, att_weight = self._mha_block(self_att_tgt,
                                               memory, memory_mask,
                                               memory_key_padding_mask)
     
        x = self.norm2(self_att_tgt + enc_att)
        x = self.norm3(x + self._ff_block(x))
        return x + tgt
        #return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)
 
    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x, att_weight = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
        return self.dropout2(x),  att_weight

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

class StackTransformer(nn.Module):
    r"""StackTransformer is a stack of N decoder layers

    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(StackTransformer, self).__init__()
        self.layers = torch.nn.modules.transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class DecoderTransformer(nn.Module):
    """
    Decoder with Transformer.
    """

    def __init__(self, encoder_dim, feature_dim, vocab_size, max_lengths, word_vocab, n_head, n_layers, dropout):
        """
        :param n_head: the number of heads in Transformer
        :param n_layers: the number of layers of Transformer
        """
        super(DecoderTransformer, self).__init__()

        # n_layers = 1
        print("decoder_n_layers=", n_layers)

        self.feature_dim = feature_dim
        self.embed_dim = feature_dim
        self.vocab_size = vocab_size
        self.max_lengths = max_lengths
        self.word_vocab = word_vocab
        self.dropout = dropout
        self.Conv1 = nn.Conv2d(encoder_dim*2, feature_dim, kernel_size = 1)
        self.LN = resblock(feature_dim, feature_dim)
        # embedding layer
        self.vocab_embedding = nn.Embedding(vocab_size, self.embed_dim)  # vocaburaly embedding
        # Transformer layer
        decoder_layer = Mesh_TransformerDecoderLayer(feature_dim, n_head, dim_feedforward=feature_dim * 4,
                                                   dropout=self.dropout)
        self.transformer = StackTransformer(decoder_layer, n_layers)
        self.position_encoding = PositionalEncoding(feature_dim, max_len=max_lengths)

        # Linear layer to find scores over vocabulary
        self.wdc = nn.Linear(feature_dim, vocab_size)
        self.dropout = nn.Dropout(p=self.dropout)
        self.cos = torch.nn.CosineSimilarity(dim=1)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence
        """
        self.vocab_embedding.weight.data.uniform_(-0.1, 0.1)

        self.wdc.bias.data.fill_(0)
        self.wdc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x1, x2, encoded_captions, caption_lengths):
        """
        :param x1, x2: encoded images, a tensor of dimension (batch_size, channel, enc_image_size, enc_image_size)
        :param encoded_captions: a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: a tensor of dimension (batch_size)
        """
        x_sam = self.cos(x1, x2)
        x = torch.cat([x1, x2], dim = 1) #+ x_sam.unsqueeze(1) #(batch_size, 2channel, enc_image_size, enc_image_size)
        x = self.LN(self.Conv1(x))

        batch, channel = x.size(0), x.size(1)
        x = x.view(batch, channel, -1).permute(2, 0, 1)
        
        word_length = encoded_captions.size(1)
        mask = torch.triu(torch.ones(word_length, word_length) * float('-inf'), diagonal=1)
        mask = mask.cuda()
        tgt_pad_mask = (encoded_captions == self.word_vocab['<NULL>'])|(encoded_captions == self.word_vocab['<END>'])

        word_emb = self.vocab_embedding(encoded_captions) #(batch, length, feature_dim)
        word_emb = word_emb.transpose(1, 0)#(length, batch, feature_dim)

        word_emb = self.position_encoding(word_emb)  # (length, batch, feature_dim)

        pred = self.transformer(word_emb, x, tgt_mask=mask, tgt_key_padding_mask=tgt_pad_mask)  # (length, batch, feature_dim)

        pred = self.wdc(self.dropout(pred))  # (length, batch, vocab_size)
        pred = pred.permute(1, 0, 2)

        # Sort input data by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoded_captions = encoded_captions[sort_ind]
        pred = pred[sort_ind]
        decode_lengths = (caption_lengths - 1).tolist()
        #encoded_caption = torch.cat((encoded_captions, torch.zeros([batch, 1], dtype = int).cuda()), dim=1)
        #decode_lengths = (caption_lengths).tolist()
        return pred, encoded_captions, decode_lengths, sort_ind

    def sample(self, x1, x2, k=1):
        """
        :param x1, x2: encoded images, a tensor of dimension (batch_size, channel, enc_image_size, enc_image_size)
        """
        x_sam = self.cos(x1, x2)
        x = torch.cat([x1, x2], dim = 1) #+ x_sam.unsqueeze(1) #(batch_size, 2channel, enc_image_size, enc_image_size)
        x = self.LN(self.Conv1(x))
        batch, channel = x.size(0), x.size(1)
        x = x.view(batch, channel, -1).permute(2, 0, 1)#(hw, batch_size, feature_dim)

        tgt = torch.zeros(batch, self.max_lengths).to(torch.int64).cuda() #(batch_size, self.max_lengths)

        mask = torch.triu(torch.ones(self.max_lengths, self.max_lengths) * float('-inf'), diagonal=1)
        mask = mask.cuda()
        tgt[:, 0] = torch.LongTensor([self.word_vocab['<START>']] *batch).cuda() #(batch_size, 1)
        seqs = torch.LongTensor([[self.word_vocab['<START>']]] *batch).cuda() #(batch_size, 1)
        #Weight = torch.zeros(1, self.max_lengths, x.size(0)).cuda()
        for step in range(self.max_lengths):
            tgt_pad_mask = (tgt == self.word_vocab['<NULL>'])
            word_emb = self.vocab_embedding(tgt)
            word_emb = word_emb.transpose(1, 0)#(length, batch, feature_dim)

            word_emb = self.position_encoding(word_emb)
            pred = self.transformer(word_emb, x, tgt_mask=mask, tgt_key_padding_mask=tgt_pad_mask)

            pred = self.wdc(self.dropout(pred))  # (length, batch, vocab_size)
            scores = pred.permute(1, 0, 2) # (batch, length, vocab_size)
            scores = scores[:, step, :].squeeze(1)  # [batch, 1, vocab_size] -> [batch, vocab_size]
            predicted_id = torch.argmax(scores, axis=-1)
            seqs = torch.cat([seqs, predicted_id.unsqueeze(1)], dim = -1)
            #Weight = torch.cat([Weight, weight], dim = 0)
            if predicted_id == self.word_vocab['<END>']:
                break
            if step<(self.max_lengths-1):#except <END> node
                tgt[:, step+1] = predicted_id
        seqs = seqs.squeeze(0)
        seqs = seqs.tolist()
        
        #feature=x.clone()
        #Weight1=Weight.clone()
        return seqs


    def sample_beam(self, x1, x2, k=1):
        """
        :param x1, x2: encoded images, a tensor of dimension (batch_size, channel, enc_image_size, enc_image_size)
        :param max_lengths: maximum length of the generated captions
        :param k: beam_size
        """

        x = torch.cat([x1, x2], dim = 1)
        x = self.LN(self.Conv1(x))
        batch, channel, h, w = x.shape
        assert batch == 1, "batch size must be 1"
        x = x.view(batch, channel, -1).unsqueeze(0).expand(k, -1, -1, -1).reshape(batch*k, channel, h*w).permute(2, 0, 1) #(h*w, batch, feature_dim)

        tgt = torch.zeros(k*batch, self.max_lengths).to(torch.int64).cuda() #(batch_size*k, self.max_lengths)

        mask = (torch.triu(torch.ones(self.max_lengths, self.max_lengths)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.cuda()
        tgt[:, 0] = torch.LongTensor([self.word_vocab['<START>']] *batch*k).cuda() #(batch_size*k, 1)
        seqs = torch.LongTensor([[self.word_vocab['<START>']]] *batch*k).cuda()
        top_k_scores = torch.zeros(k*batch, 1).cuda()
        complete_seqs = []
        complete_seqs_scores = []
        for step in range(self.max_lengths):
            word_emb = self.vocab_embedding(tgt)
            word_emb = word_emb.transpose(1, 0)
            word_emb = self.position_encoding(word_emb)
            pred = self.transformer(word_emb, x, tgt_mask=mask)
            pred = self.wdc(self.dropout(pred))  # (length, batch, vocab_size)
            scores = pred.permute(1, 0, 2) # (batch, length, vocab_size)
            scores = scores[:, step, :].squeeze(1)  # [batch, 1, vocab_size] -> [batch, vocab_size]
            scores = F.log_softmax(scores, dim=1)
            scores = top_k_scores.expand_as(scores) + scores
            if step == 0:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            # prev_word_inds = top_k_words // vocab_size  # (s)
            prev_word_inds = torch.div(top_k_words, self.vocab_size, rounding_mode='floor')
            next_word_inds = top_k_words % self.vocab_size  # (s)
            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim = 1)
            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != self.word_vocab['<END>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            x = x[:,prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            tgt = tgt[incomplete_inds]
            if step<self.max_lengths-1:
                tgt[:, :step+2] = seqs


        if complete_seqs == []:
            complete_seqs.extend(seqs[incomplete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[incomplete_inds])
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        return seq


    def fine_tune(self, fine_tune=True):
        for p in self.parameters():
            p.requires_grad = fine_tune


class DecoderTransformerV1(nn.Module):
    """
    Decoder with Transformer.
    """

    def __init__(self, encoder_dim, feature_dim, vocab_size, max_lengths, word_vocab, n_head, n_layers, dropout):
        """
        :param n_head: the number of heads in Transformer
        :param n_layers: the number of layers of Transformer
        """
        super(DecoderTransformerV1, self).__init__()

        # n_layers = 1
        print("decoder_n_layers=", n_layers)

        self.feature_dim = feature_dim
        self.embed_dim = feature_dim
        self.vocab_size = vocab_size
        self.max_lengths = max_lengths
        self.word_vocab = word_vocab
        self.dropout = dropout
        self.Conv1 = nn.Conv2d(encoder_dim * 2, feature_dim, kernel_size=1)
        self.LN = resblock(feature_dim, feature_dim)
        # embedding layer
        self.vocab_embedding = nn.Embedding(vocab_size, self.embed_dim)  # vocaburaly embedding
        # Transformer layer
        decoder_layer = Mesh_TransformerDecoderLayer(feature_dim, n_head, dim_feedforward=feature_dim * 4,
                                                     dropout=self.dropout)
        self.transformer = StackTransformer(decoder_layer, n_layers)
        self.position_encoding = PositionalEncoding(feature_dim, max_len=max_lengths)

        # Linear layer to find scores over vocabulary
        self.wdc = nn.Linear(feature_dim, vocab_size)
        self.dropout = nn.Dropout(p=self.dropout)
        self.cos = torch.nn.CosineSimilarity(dim=1)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence
        """
        self.vocab_embedding.weight.data.uniform_(-0.1, 0.1)

        self.wdc.bias.data.fill_(0)
        self.wdc.weight.data.uniform_(-0.1, 0.1)

    def forward(
            self,
            x1, x2,  # (B, 64, 8, 8)
            encoded_captions, caption_lengths,
            seg_pre=None,  # (B, 3, 256, 256)  来自分割分支
            is_logits=True,  # seg_pre 是否是 logits
            token_mode="per_class",  # "single" 或 "per_class"
            change_idx=(1, 2),  # 哪些通道属于“变化类”
            detach_mask=True,  # 先冻结 mask 梯度更稳
            gamma=1.0  # 锐化系数: m**gamma, 1.0 表示不变
    ):
        """
        :param x1, x2: encoded images, a tensor of dimension (batch_size, channel, enc_image_size, enc_image_size)
        :param encoded_captions: a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: a tensor of dimension (batch_size)
        """
        x = torch.cat([x1, x2], dim=1)  # (B, 128, 8, 8)
        x = self.LN(self.Conv1(x))  # (B, Cv, 8, 8)
        B, Cv, H, W = x.size()
        mem = x.view(B, Cv, -1).permute(2, 0, 1)  # (H*W=64, B, Cv)

        if seg_pre is not None:
            # 2.1 seg_pre -> 概率  (多类分割用 softmax)
            prob = torch.softmax(seg_pre, dim=1) if is_logits else seg_pre.clamp(0, 1)

            # 2.2 下采样到与 x 一样的空间分辨率
            prob8 = torch.nn.functional.adaptive_avg_pool2d(prob, (H, W))  # (B, 3, 8, 8)

            # 2.3 生成 1 或 多个 change token
            tokens = []
            if token_mode == "single":
                # 将所有变化类合成 1 个前景权重图
                w = prob8[:, list(change_idx), :, :].sum(dim=1, keepdim=True)  # (B,1,8,8)
                if gamma != 1.0:
                    w = w.clamp_min(1e-6).pow(gamma)
                if detach_mask:
                    w = w.detach()
                num = (x * w).sum(dim=(-1, -2))  # (B, Cv)
                den = w.sum(dim=(-1, -2)).clamp_min(1e-6)  # (B, 1)
                v = (num / den).unsqueeze(0)  # (1, B, Cv)
                tokens.append(v)
            else:  # "per_class"
                for cls in change_idx:
                    w = prob8[:, cls:cls + 1, :, :]  # (B,1,8,8)
                    if gamma != 1.0:
                        w = w.clamp_min(1e-6).pow(gamma)
                    if detach_mask:
                        w = w.detach()
                    num = (x * w).sum(dim=(-1, -2))  # (B, Cv)
                    den = w.sum(dim=(-1, -2)).clamp_min(1e-6)  # (B, 1)
                    v = (num / den).unsqueeze(0)  # (1, B, Cv)
                    tokens.append(v)

            if tokens:
                mem = torch.cat([mem] + tokens, dim=0)  # (64 + T, B, Cv), T=1 或 len(change_idx)

        word_length = encoded_captions.size(1)
        mask = torch.triu(torch.ones(word_length, word_length) * float('-inf'), diagonal=1)
        mask = mask.cuda()
        tgt_pad_mask = (encoded_captions == self.word_vocab['<NULL>']) | (encoded_captions == self.word_vocab['<END>'])

        word_emb = self.vocab_embedding(encoded_captions)  # (batch, length, feature_dim)
        word_emb = word_emb.transpose(1, 0)  # (length, batch, feature_dim)

        word_emb = self.position_encoding(word_emb)  # (length, batch, feature_dim)

        pred = self.transformer(word_emb, mem, tgt_mask=mask,
                                tgt_key_padding_mask=tgt_pad_mask)  # (length, batch, feature_dim)

        pred = self.wdc(self.dropout(pred))  # (length, batch, vocab_size)
        pred = pred.permute(1, 0, 2)

        # Sort input data by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoded_captions = encoded_captions[sort_ind]
        pred = pred[sort_ind]
        decode_lengths = (caption_lengths - 1).tolist()
        # encoded_caption = torch.cat((encoded_captions, torch.zeros([batch, 1], dtype = int).cuda()), dim=1)
        # decode_lengths = (caption_lengths).tolist()
        return pred, encoded_captions, decode_lengths, sort_ind

    def get_last_hidden_from_inputs(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            token: torch.Tensor,  # (B, L) 目标序列（teacher forcing）
            *,
            seg_pre: torch.Tensor = None,  # (B, Cs, Hs, Ws) 分割支路输出（logits 或 概率）
            is_logits: bool = True,  # seg_pre 是否为 logits
            token_mode: str = "per_class",  # "single" 或 "per_class"
            change_idx: tuple = (1, 2),  # 哪些通道属于“变化类”
            detach_mask: bool = True,  # 是否对 change 权重图 stop-grad
            gamma: float = 1.0,  # 权重锐化 w <- w^gamma
            detach: bool = True,  # 是否对返回的隐藏态 stop-grad（给 Critic 推荐 True）
            build_pad_from_vocab: bool = True,  # 用词表里的 <NULL>/<END> 构造 padding mask
            mem_append_tokens: bool = True  # 是否把 change token 拼接到 memory 尾部
    ) -> torch.Tensor:
        """
        在解码器内部构造视觉 memory / 文本 mask / change tokens，并返回 Transformer
        最后一层隐藏状态 (L, B, D)。该方法不计算 logits，也不改变模型其他参数。

        返回:
            last_hidden: torch.Tensor, 形状 (L, B, D)
        """
        assert hasattr(self, "Conv1") and hasattr(self, "LN")
        assert hasattr(self, "vocab_embedding") and hasattr(self, "position_encoding")
        assert hasattr(self, "transformer")

        device = x1.device
        # -----------------------------
        # 1) 视觉侧：构造 memory（含可选 change token）
        # -----------------------------
        # Backbone 融合 + 1x1 投影 + 残块
        x = torch.cat([x1, x2], dim=1)  # (B, 2*enc_dim, H8, W8)
        x = self.LN(self.Conv1(x))  # (B, Cv, H8, W8)
        B, Cv, H, W = x.shape

        # 基础 memory（未拼接 change token）
        mem = x.view(B, Cv, -1).permute(2, 0, 1).contiguous()  # (H*W, B, Cv)

        # 可选：由分割图构造 change token 并拼接到 memory 尾部
        if seg_pre is not None and mem_append_tokens:
            seg_pre = seg_pre.to(device)
            # 2.1 seg_pre -> 概率
            if is_logits:
                prob = torch.softmax(seg_pre, dim=1)  # (B, Cs, Hs, Ws)
            else:
                prob = seg_pre.clamp(0, 1)

            # 2.2 下采样到与 x 一样的空间分辨率
            prob8 = torch.nn.functional.adaptive_avg_pool2d(prob, (H, W))  # (B, Cs, H, W)

            # 2.3 生成单/多 change token
            tokens = []
            if token_mode == "single":
                # 所有变化类合成一个权重图
                w = prob8[:, list(change_idx), :, :].sum(dim=1, keepdim=True)  # (B,1,H,W)
                if gamma != 1.0:
                    w = w.clamp_min(1e-6).pow(gamma)
                if detach_mask:
                    w = w.detach()
                num = (x * w).sum(dim=(-1, -2))  # (B, Cv)
                den = w.sum(dim=(-1, -2)).clamp_min(1e-6)  # (B, 1)
                v = (num / den).unsqueeze(0)  # (1, B, Cv)
                tokens.append(v)
            else:  # "per_class"
                for cls in change_idx:
                    w = prob8[:, cls:cls + 1, :, :]  # (B,1,H,W)
                    if gamma != 1.0:
                        w = w.clamp_min(1e-6).pow(gamma)
                    if detach_mask:
                        w = w.detach()
                    num = (x * w).sum(dim=(-1, -2))  # (B, Cv)
                    den = w.sum(dim=(-1, -2)).clamp_min(1e-6)  # (B, 1)
                    v = (num / den).unsqueeze(0)  # (1, B, Cv)
                    tokens.append(v)

            if tokens:
                mem = torch.cat([mem] + tokens, dim=0)  # (H*W + T, B, Cv)

        # -----------------------------
        # 2) 文本侧：嵌入 + 位置编码 + mask
        # -----------------------------
        # token: (B, L) → (L, B, D)
        word_emb = self.vocab_embedding(token.to(device))  # (B, L, D)
        word_emb = word_emb.transpose(1, 0).contiguous()  # (L, B, D)
        word_emb = self.position_encoding(word_emb)  # (L, B, D)

        L = word_emb.size(0)

        # 自回归上三角 mask（未来时刻不可见）
        # 浮点型，非对角线处设 -inf
        attn_mask = torch.triu(
            torch.ones(L, L, device=device) * float('-inf'),
            diagonal=1
        )

        # padding mask: (B, L) 的 bool，True 表示需要 mask
        if build_pad_from_vocab:
            assert hasattr(self, "word_vocab") and isinstance(self.word_vocab, dict)
            pad_id = self.word_vocab.get('<NULL>', None)
            end_id = self.word_vocab.get('<END>', None)
            if pad_id is None:
                raise ValueError("word_vocab 缺少 '<NULL>'，无法构造 padding mask")
            tgt_pad_mask = (token == pad_id)
            if end_id is not None:
                tgt_pad_mask = tgt_pad_mask | (token == end_id)
        else:
            # 若外部已构造，可改为入参传入；此处给出兜底
            tgt_pad_mask = torch.zeros(token.shape[0], token.shape[1], dtype=torch.bool, device=device)

        # -----------------------------
        # 3) Transformer 解码：返回最后一层隐藏态
        # -----------------------------
        # 注意：你的 Mesh_TransformerDecoderLayer/StackTransformer 接口与 forward 一致：
        # pred = self.transformer(word_emb, mem, tgt_mask=..., tgt_key_padding_mask=...)
        last_hidden = self.transformer(
            word_emb,  # (L, B, D)
            mem,  # (H*W [+T], B, Cv)
            tgt_mask=attn_mask,  # (L, L) float, 上三角 -inf
            tgt_key_padding_mask=tgt_pad_mask  # (B, L) bool
        )  # -> (L, B, D)

        if detach:
            last_hidden = last_hidden.detach()

        return last_hidden  # (L, B, D)

    @torch.no_grad()
    def sample(self, x1, x2, k=1, seg_pre=None, is_logits=True,
               token_mode="per_class", change_idx=(1, 2), gamma=1.0):
        """
        Greedy 解码；支持将 seg_pre 生成的 change token 拼到 memory 尾部
        """
        # 1) 视觉编码
        x = torch.cat([x1, x2], dim=1)  # (B, 128, 8, 8)
        x = self.LN(self.Conv1(x))  # (B, Cv, 8, 8)
        B, Cv, H, W = x.size()
        mem = x.view(B, Cv, -1).permute(2, 0, 1)  # (H*W, B, Cv)

        # 2) seg_pre -> change token(s)
        if seg_pre is not None:
            seg_pre = seg_pre.to(x.device)
            prob = torch.softmax(seg_pre, dim=1) if is_logits else seg_pre.clamp(0, 1)
            prob8 = F.adaptive_avg_pool2d(prob, (H, W))  # (B, Cs, 8, 8)

            tokens = []
            if token_mode == "single":
                w = prob8[:, list(change_idx), :, :].sum(1, keepdim=True)  # (B,1,8,8)
                if gamma != 1.0: w = w.clamp_min(1e-6).pow(gamma)
                num = (x * w).sum(dim=(-1, -2))  # (B, Cv)
                den = w.sum(dim=(-1, -2))  # (B, 1)
                empty = (den < 1e-6)
                den = den.clamp_min(1e-6)
                v = (num / den)  # (B, Cv)
                if empty.any():
                    v_fallback = x.mean(dim=(-1, -2))  # (B, Cv)
                    v[empty.squeeze(1)] = v_fallback[empty.squeeze(1)]
                tokens.append(v.unsqueeze(0))  # (1, B, Cv)
            else:  # per_class
                for cls in change_idx:
                    w = prob8[:, cls:cls + 1, :, :]  # (B,1,8,8)
                    if gamma != 1.0: w = w.clamp_min(1e-6).pow(gamma)
                    num = (x * w).sum(dim=(-1, -2))
                    den = w.sum(dim=(-1, -2))
                    empty = (den < 1e-6)
                    den = den.clamp_min(1e-6)
                    v = (num / den)
                    if empty.any():
                        v_fallback = x.mean(dim=(-1, -2))
                        v[empty.squeeze(1)] = v_fallback[empty.squeeze(1)]
                    tokens.append(v.unsqueeze(0))
            if tokens:
                mem = torch.cat([mem] + tokens, dim=0)  # (H*W+T, B, Cv)

        # 3) 自回归解码（greedy）
        L = self.max_lengths
        tgt = torch.zeros(B, L, dtype=torch.int64, device=x.device)
        mask = torch.triu(torch.ones(L, L, device=x.device) * float('-inf'), diagonal=1)
        tgt[:, 0] = self.word_vocab['<START>']
        seqs = torch.tensor([[self.word_vocab['<START>']]] * B, device=x.device)

        for step in range(L):
            tgt_pad_mask = (tgt == self.word_vocab['<NULL>'])
            word_emb = self.position_encoding(self.vocab_embedding(tgt).transpose(1, 0))
            pred = self.transformer(word_emb, mem, tgt_mask=mask, tgt_key_padding_mask=tgt_pad_mask)
            pred = self.wdc(self.dropout(pred))  # (L, B, V)
            scores = pred.permute(1, 0, 2)[:, step, :]  # (B, V)
            predicted_id = torch.argmax(scores, dim=-1)  # (B,)
            seqs = torch.cat([seqs, predicted_id.unsqueeze(1)], dim=-1)
            if (predicted_id == self.word_vocab['<END>']).all():
                break
            if step < L - 1:
                tgt[:, step + 1] = predicted_id

        return seqs.squeeze(0).tolist() if B == 1 else [s.tolist() for s in seqs]

    @torch.no_grad()
    def sample_beam(self, x1, x2, k=1, seg_pre=None, is_logits=True,
                    token_mode="per_class", change_idx=(1, 2), gamma=1.0):
        """
        Beam Search 解码（batch==1）；同样支持 change token
        """
        x = torch.cat([x1, x2], dim=1)
        x = self.LN(self.Conv1(x))  # (1, Cv, 8, 8)
        batch, channel, H, W = x.shape
        assert batch == 1, "batch size must be 1"

        # 1) base memory（未扩 beam）
        mem = x.view(batch, channel, -1).permute(2, 0, 1)  # (H*W, 1, Cv)

        # 2) seg_pre -> change token(s)
        if seg_pre is not None:
            seg_pre = seg_pre.to(x.device)
            prob = torch.softmax(seg_pre, dim=1) if is_logits else seg_pre.clamp(0, 1)
            prob8 = F.adaptive_avg_pool2d(prob, (H, W))  # (1, Cs, 8, 8)

            tokens = []
            if token_mode == "single":
                w = prob8[:, list(change_idx), :, :].sum(1, keepdim=True)  # (1,1,8,8)
                if gamma != 1.0: w = w.clamp_min(1e-6).pow(gamma)
                num = (x * w).sum(dim=(-1, -2))  # (1, Cv)
                den = w.sum(dim=(-1, -2))
                empty = (den < 1e-6)
                den = den.clamp_min(1e-6)
                v = (num / den)  # (1, Cv)
                if empty.any():
                    v_fallback = x.mean(dim=(-1, -2))
                    v[empty.squeeze(1)] = v_fallback[empty.squeeze(1)]
                tokens.append(v.unsqueeze(0))  # (1,1,Cv)
            else:
                for cls in change_idx:
                    w = prob8[:, cls:cls + 1, :, :]
                    if gamma != 1.0: w = w.clamp_min(1e-6).pow(gamma)
                    num = (x * w).sum(dim=(-1, -2))
                    den = w.sum(dim=(-1, -2))
                    empty = (den < 1e-6)
                    den = den.clamp_min(1e-6)
                    v = (num / den)
                    if empty.any():
                        v_fallback = x.mean(dim=(-1, -2))
                        v[empty.squeeze(1)] = v_fallback[empty.squeeze(1)]
                    tokens.append(v.unsqueeze(0))
            if tokens:
                mem = torch.cat([mem] + tokens, dim=0)  # (H*W+T, 1, Cv)

        # 3) 扩 beam（第二维）
        mem = mem.expand(-1, k, -1)  # (H*W+T, k, Cv)

        # 4) beam search
        L = self.max_lengths
        tgt = torch.zeros(k * batch, L, dtype=torch.int64, device=x.device)
        mask = (torch.triu(torch.ones(L, L, device=x.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(~mask, float('-inf')).masked_fill(mask, 0.0)

        tgt[:, 0] = self.word_vocab['<START>']
        seqs = torch.tensor([[self.word_vocab['<START>']]] * (batch * k), device=x.device)
        top_k_scores = torch.zeros(k * batch, 1, device=x.device)
        complete_seqs, complete_seqs_scores = [], []

        for step in range(L):
            word_emb = self.position_encoding(self.vocab_embedding(tgt).transpose(1, 0))
            pred = self.transformer(word_emb, mem, tgt_mask=mask)  # (L, k, C)
            pred = self.wdc(self.dropout(pred))
            scores = F.log_softmax(pred.permute(1, 0, 2)[:, step, :], dim=1)  # (k, V)
            scores = top_k_scores.expand_as(scores) + scores

            if step == 0:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

            prev_word_inds = torch.div(top_k_words, self.vocab_size, rounding_mode='floor')
            next_word_inds = top_k_words % self.vocab_size
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

            incomplete_inds = [ind for ind, w in enumerate(next_word_inds) if w != self.word_vocab['<END>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds].view(-1).tolist())

            k -= len(complete_inds)
            if k == 0:
                break

            # 收缩 beam（同步重排 memory / 分数 / tgt / seqs）
            seqs = seqs[incomplete_inds]
            mem = mem[:, prev_word_inds[incomplete_inds], :]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            tgt = tgt[incomplete_inds]
            if step < L - 1:
                tgt[:, :step + 2] = seqs

        if complete_seqs == []:
            complete_seqs.extend(seqs.tolist())
            complete_seqs_scores.extend(top_k_scores.squeeze(1).tolist())

        i = int(torch.tensor(complete_seqs_scores).argmax().item())
        return complete_seqs[i]

    def fine_tune(
            self,
            fine_tune: bool = True,
            *,
            freeze_embeddings: bool = False,  # 冻结 vocab_embedding
            freeze_posenc: bool = False,  # 冻结位置编码模块（若存在）
            freeze_visual_proj: bool = False,  # 冻结视觉侧 Conv1/LN（若存在）
            only_head: bool = False,  # 只训练输出头 wdc
            unfreeze_last_n: int = None,  # 只解冻解码器最后 N 层（其余层冻结）
            verbose: bool = False
    ):
        """
        更细粒度的微调控制：
        - fine_tune=False: 全部冻结
        - only_head=True: 只训练输出头（wdc），其余冻结
        - unfreeze_last_n: 只解冻 transformer 的最后 N 层
        - freeze_embeddings/posenc/visual_proj: 定点冻结这些模块
        """
        # 1) 默认先统一开/关
        for p in self.parameters():
            p.requires_grad = fine_tune

        # 2) 只训练输出头（优先级最高）
        if only_head:
            for name, p in self.named_parameters():
                p.requires_grad = ('wdc' in name)  # 只留输出线性层
            if verbose: print('[decoder] only_head=True -> train wdc only')
            return

        # 3) 定点冻结：词嵌入 / 位置编码 / 视觉投影
        if freeze_embeddings and hasattr(self, 'vocab_embedding'):
            for p in self.vocab_embedding.parameters():
                p.requires_grad = False
            if verbose: print('[decoder] freeze vocab_embedding')

        if freeze_posenc and hasattr(self, 'position_encoding'):
            for p in self.position_encoding.parameters():
                p.requires_grad = False
            if verbose: print('[decoder] freeze position_encoding')

        if freeze_visual_proj:
            # 视觉侧投影：LN/Conv1（若存在就冻上）
            if hasattr(self, 'Conv1'):
                for p in self.Conv1.parameters(): p.requires_grad = False
            if hasattr(self, 'LN') and isinstance(self.LN, nn.Module):
                for p in self.LN.parameters(): p.requires_grad = False
            if verbose: print('[decoder] freeze visual projection (Conv1/LN)')

        # 4) 只解冻最后 N 层解码器
        if unfreeze_last_n is not None and unfreeze_last_n >= 0:
            # 尝试找到 decoder 层列表（适配 nn.TransformerDecoder 或自定义）
            layers = []
            if hasattr(self, 'transformer') and hasattr(self.transformer, 'layers'):
                layers = list(self.transformer.layers)
            elif hasattr(self, 'transformer') and hasattr(self.transformer, 'decoder') and hasattr(
                    self.transformer.decoder, 'layers'):
                layers = list(self.transformer.decoder.layers)

            if layers:
                # 先冻住所有层
                for l in layers:
                    for p in l.parameters(): p.requires_grad = False
                # 再只解冻最后 N 层
                if unfreeze_last_n > 0:
                    for l in layers[-unfreeze_last_n:]:
                        for p in l.parameters(): p.requires_grad = True
                if verbose:
                    total = len(layers)
                    print(f'[decoder] unfreeze_last_n={unfreeze_last_n} / total_layers={total}')


class DecoderTransformer2(nn.Module):
    """
    Decoder with Transformer.
    """

    def __init__(self, encoder_dim, feature_dim, vocab_size, max_lengths, word_vocab, n_head, n_layers, dropout):
        """
        :param n_head: the number of heads in Transformer
        :param n_layers: the number of layers of Transformer
        """
        super(DecoderTransformer2, self).__init__()

        # n_layers = 1
        print("decoder_n_layers=", n_layers)

        self.feature_dim = feature_dim
        self.embed_dim = feature_dim
        self.vocab_size = vocab_size
        self.max_lengths = max_lengths
        self.word_vocab = word_vocab
        self.dropout = dropout
        self.Conv1 = nn.Conv2d(encoder_dim * 2, feature_dim, kernel_size=1)
        self.LN = resblock(feature_dim, feature_dim)
        # embedding layer
        self.vocab_embedding = nn.Embedding(vocab_size, self.embed_dim)  # vocaburaly embedding
        # Transformer layer
        decoder_layer = Mesh_TransformerDecoderLayer(feature_dim, n_head, dim_feedforward=feature_dim * 4,
                                                     dropout=self.dropout)
        self.transformer = StackTransformer(decoder_layer, n_layers)
        self.position_encoding = PositionalEncoding(feature_dim, max_len=max_lengths)

        # Linear layer to find scores over vocabulary
        self.wdc = nn.Linear(feature_dim, vocab_size)
        self.dropout = nn.Dropout(p=self.dropout)
        self.cos = torch.nn.CosineSimilarity(dim=1)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence
        """
        self.vocab_embedding.weight.data.uniform_(-0.1, 0.1)

        self.wdc.bias.data.fill_(0)
        self.wdc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x1, x2, encoded_captions, caption_lengths):
        """
        :param x1, x2: encoded images, a tensor of dimension (batch_size, channel, enc_image_size, enc_image_size)
        :param encoded_captions: a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: a tensor of dimension (batch_size)
        """
        x_sam = self.cos(x1, x2)
        x = torch.cat([x1, x2], dim=1)  # + x_sam.unsqueeze(1) #(batch_size, 2channel, enc_image_size, enc_image_size)
        x = self.LN(self.Conv1(x))

        batch, channel = x.size(0), x.size(1)
        x = x.view(batch, channel, -1).permute(2, 0, 1)

        word_length = encoded_captions.size(1)
        mask = torch.triu(torch.ones(word_length, word_length) * float('-inf'), diagonal=1)
        mask = mask.cuda()
        tgt_pad_mask = (encoded_captions == self.word_vocab['<NULL>']) | (encoded_captions == self.word_vocab['<END>'])

        word_emb = self.vocab_embedding(encoded_captions)  # (batch, length, feature_dim)
        word_emb = word_emb.transpose(1, 0)  # (length, batch, feature_dim)

        word_emb = self.position_encoding(word_emb)  # (length, batch, feature_dim)

        pred = self.transformer(word_emb, x, tgt_mask=mask,
                                tgt_key_padding_mask=tgt_pad_mask)  # (length, batch, feature_dim)
        # 新增
        cc_features = pred
        pred = self.wdc(self.dropout(pred))  # (length, batch, vocab_size)
        pred = pred.permute(1, 0, 2)

        # Sort input data by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoded_captions = encoded_captions[sort_ind]
        pred = pred[sort_ind]
        decode_lengths = (caption_lengths - 1).tolist()
        # encoded_caption = torch.cat((encoded_captions, torch.zeros([batch, 1], dtype = int).cuda()), dim=1)
        # decode_lengths = (caption_lengths).tolist()
        return pred, encoded_captions, decode_lengths, sort_ind, cc_features

    def sample(self, x1, x2, k=1):
        """
        :param x1, x2: encoded images, a tensor of dimension (batch_size, channel, enc_image_size, enc_image_size)
        """
        x_sam = self.cos(x1, x2)
        x = torch.cat([x1, x2], dim=1)  # + x_sam.unsqueeze(1) #(batch_size, 2channel, enc_image_size, enc_image_size)
        x = self.LN(self.Conv1(x))
        batch, channel = x.size(0), x.size(1)
        x = x.view(batch, channel, -1).permute(2, 0, 1)  # (hw, batch_size, feature_dim)

        tgt = torch.zeros(batch, self.max_lengths).to(torch.int64).cuda()  # (batch_size, self.max_lengths)

        mask = torch.triu(torch.ones(self.max_lengths, self.max_lengths) * float('-inf'), diagonal=1)
        mask = mask.cuda()
        tgt[:, 0] = torch.LongTensor([self.word_vocab['<START>']] * batch).cuda()  # (batch_size, 1)
        seqs = torch.LongTensor([[self.word_vocab['<START>']]] * batch).cuda()  # (batch_size, 1)
        # Weight = torch.zeros(1, self.max_lengths, x.size(0)).cuda()
        for step in range(self.max_lengths):
            tgt_pad_mask = (tgt == self.word_vocab['<NULL>'])
            word_emb = self.vocab_embedding(tgt)
            word_emb = word_emb.transpose(1, 0)  # (length, batch, feature_dim)

            word_emb = self.position_encoding(word_emb)
            pred = self.transformer(word_emb, x, tgt_mask=mask, tgt_key_padding_mask=tgt_pad_mask)

            pred = self.wdc(self.dropout(pred))  # (length, batch, vocab_size)
            scores = pred.permute(1, 0, 2)  # (batch, length, vocab_size)
            scores = scores[:, step, :].squeeze(1)  # [batch, 1, vocab_size] -> [batch, vocab_size]
            predicted_id = torch.argmax(scores, axis=-1)
            seqs = torch.cat([seqs, predicted_id.unsqueeze(1)], dim=-1)
            # Weight = torch.cat([Weight, weight], dim = 0)
            if predicted_id == self.word_vocab['<END>']:
                break
            if step < (self.max_lengths - 1):  # except <END> node
                tgt[:, step + 1] = predicted_id
        seqs = seqs.squeeze(0)
        seqs = seqs.tolist()

        # feature=x.clone()
        # Weight1=Weight.clone()
        return seqs

    def sample_beam(self, x1, x2, k=1):
        """
        :param x1, x2: encoded images, a tensor of dimension (batch_size, channel, enc_image_size, enc_image_size)
        :param max_lengths: maximum length of the generated captions
        :param k: beam_size
        """

        x = torch.cat([x1, x2], dim=1)
        x = self.LN(self.Conv1(x))
        batch, channel, h, w = x.shape
        assert batch == 1, "batch size must be 1"
        x = x.view(batch, channel, -1).unsqueeze(0).expand(k, -1, -1, -1).reshape(batch * k, channel, h * w).permute(2,
                                                                                                                     0,
                                                                                                                     1)  # (h*w, batch, feature_dim)

        tgt = torch.zeros(k * batch, self.max_lengths).to(torch.int64).cuda()  # (batch_size*k, self.max_lengths)

        mask = (torch.triu(torch.ones(self.max_lengths, self.max_lengths)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.cuda()
        tgt[:, 0] = torch.LongTensor([self.word_vocab['<START>']] * batch * k).cuda()  # (batch_size*k, 1)
        seqs = torch.LongTensor([[self.word_vocab['<START>']]] * batch * k).cuda()
        top_k_scores = torch.zeros(k * batch, 1).cuda()
        complete_seqs = []
        complete_seqs_scores = []
        for step in range(self.max_lengths):
            word_emb = self.vocab_embedding(tgt)
            word_emb = word_emb.transpose(1, 0)
            word_emb = self.position_encoding(word_emb)
            pred = self.transformer(word_emb, x, tgt_mask=mask)
            pred = self.wdc(self.dropout(pred))  # (length, batch, vocab_size)
            scores = pred.permute(1, 0, 2)  # (batch, length, vocab_size)
            scores = scores[:, step, :].squeeze(1)  # [batch, 1, vocab_size] -> [batch, vocab_size]
            scores = F.log_softmax(scores, dim=1)
            scores = top_k_scores.expand_as(scores) + scores
            if step == 0:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            # prev_word_inds = top_k_words // vocab_size  # (s)
            prev_word_inds = torch.div(top_k_words, self.vocab_size, rounding_mode='floor')
            next_word_inds = top_k_words % self.vocab_size  # (s)
            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != self.word_vocab['<END>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            x = x[:, prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            tgt = tgt[incomplete_inds]
            if step < self.max_lengths - 1:
                tgt[:, :step + 2] = seqs

        if complete_seqs == []:
            complete_seqs.extend(seqs[incomplete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[incomplete_inds])
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        return seq

    def fine_tune(self, fine_tune=True):
        for p in self.parameters():
            p.requires_grad = fine_tune


