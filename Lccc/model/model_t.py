import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
import math
# 1. Multi-Scale Attention Suppression with ReLU and Dropout.
class MultiScaleAttentionSuppression(nn.Module):
    """
    Bidirectional attention suppression module for multi-scale features with ReLU and Dropout after output projection.

    Args:
        t1_channels (list) : [64, 128, 320, 512], channel counts for multi-scale T1 features.
        t2_channels (int)  : 512, channel count for T2 features.
        proj_dim (int)     : 256, shared intermediate projection dimension.
        learnable_alpha    : whether alpha is learnable.
        init_alpha (float) : initial alpha value.
    """
    def __init__(
        self,
        t1_channels=(64, 128, 320, 512),
        t2_channels=512,
        proj_dim=256,
        learnable_alpha=True,
        init_alpha=0.1,
    ):
        super().__init__()
        self.proj_dim = proj_dim

        # -- 1) Project all features to proj_dim. --
        self.t1_projs = nn.ModuleList([nn.Conv2d(c, proj_dim, 1) for c in t1_channels])
        self.t2_proj  = nn.Conv2d(t2_channels, proj_dim, 1)

        # -- 2) Q / K / V linear layers. --
        self.query_transform = nn.Linear(proj_dim, proj_dim)
        self.key_transform   = nn.Linear(proj_dim, proj_dim)
        self.value_transform = nn.Linear(proj_dim, proj_dim)

        # -- 3) Output projection: Conv -> ReLU -> Dropout. --
        self.t1_out_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(proj_dim, c, 1),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1)
            ) for c in t1_channels
        ])
        self.t2_out_proj = nn.Sequential(
            nn.Conv2d(proj_dim, t2_channels, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1)
        )

        # -- 4) Learnable or fixed alpha. --
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(init_alpha))
        else:
            self.register_buffer('alpha', torch.tensor(init_alpha))

    # --------- Helper functions ---------
    @staticmethod
    def _reshape_feat(x):
        """Reshape [B,C,H,W] to [B,HW,C]."""
        B, C, H, W = x.shape
        return x.view(B, C, -1).permute(0, 2, 1)

    def _attention(self, q, k):
        """Scaled dot-product attention returning weights with shape [B,Lq,Lk]."""
        q = self.query_transform(q)
        k = self.key_transform(k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.proj_dim)
        return F.softmax(scores, dim=-1)

    # --------- Forward pass ---------
    def forward(self, img1_feats, img2_feats):
        """
        img*_feats: list of 5 feature scales:
                    [CD_s1, CD_s2, CD_s3, CD_s4, CC_feature]
        Returns:
        - sup_img1_feats, sup_img2_feats: suppressed feature lists with the same format as the input.
        """
        # 1) Split CD and CC features.
        CD1, CC1 = img1_feats[:-1], img1_feats[-1]
        CD2, CC2 = img2_feats[:-1], img2_feats[-1]

        CD1_sup, CC1_sup = self._process_branch(CD1, CC1)
        CD2_sup, CC2_sup = self._process_branch(CD2, CC2)

        return CD1_sup + [CC1_sup], CD2_sup + [CC2_sup]
        # return CD1_sup + [CC1], CD2_sup + [CC2]

    # ------------ Private branch function -------------
    def _process_branch(self, t1_feats, t2_feat):
        """
        t1_feats: 4 x [B,C,H,W]; t2_feat: [B,512,8,8]
        Returns suppressed features with the same structure.
        """
        B = t1_feats[0].shape[0]

        # 1) Project with 1x1 convolutions to the shared dimension.
        proj_t1 = [proj(f) for proj, f in zip(self.t1_projs, t1_feats)]
        proj_t2 = self.t2_proj(t2_feat)

        # 2) Flatten features for attention.
        flat_t1 = [self._reshape_feat(f) for f in proj_t1]  # list[B, HW, C]
        flat_t2 = self._reshape_feat(proj_t2)               # [B, HW, C]

        # 3) Compute T1->T2 and T2->T1 attention for each scale.
        sup_t1 = []
        A_2to1_all = []
        for i, f1 in enumerate(flat_t1):
            A_1to2 = self._attention(f1, flat_t2)  # [B,L1,L2]
            A_2to1 = self._attention(flat_t2, f1)  # [B,L2,L1]
            A_2to1_all.append(A_2to1)

            # Suppression mask.
            mask1 = 1.0 - self.alpha * A_1to2.mean(dim=-1, keepdim=True)
            f1_sup = (f1 * mask1).permute(0, 2, 1).view(B, self.proj_dim, *t1_feats[i].shape[2:])
            # Output projection + ReLU + Dropout.
            sup_t1.append(self.t1_out_projs[i](f1_sup))

        # 4) Suppress T2 by averaging A_2to1 over all scales.
        t2_mask = 1.0 - self.alpha * torch.stack(
            [A.mean(dim=-1, keepdim=True) for A in A_2to1_all], dim=0
        ).mean(dim=0)  # [B,L2,1]

        f2_sup = (flat_t2 * t2_mask).permute(0, 2, 1).view(B, self.proj_dim, *t2_feat.shape[2:])
        sup_t2 = self.t2_out_proj(f2_sup)  # Conv -> ReLU -> Dropout

        return sup_t1, sup_t2



class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, multi_in_channels=[64, 128, 320, 512], context_in_channels=512):
        """
        Initialize the fusion module.
        multi_in_channels: list of channel counts for the multi-scale feature maps (prior to context concat).
        context_in_channels: channel count for the context feature (last feature map).
        """
        super(MultiScaleFeatureFusion, self).__init__()
        self.multi_in_channels = multi_in_channels
        self.num_scales = len(multi_in_channels)
        # Compute output channel (embed) for each scale after context concatenation
        self.embed_dims = [m_ch + context_in_channels for m_ch in multi_in_channels]

        self.context_fc = nn.Sequential(
            nn.Linear(context_in_channels, context_in_channels),
            nn.ReLU(inplace=True),
        )

        # Define fusion layers for each scale (inspired by FusDiff composition)
        # Each scale has:
        #  - fc1: (embed*2 -> embed), fc2: (embed*2 -> embed), each followed by ReLU
        #  - fc_final: (embed*3 -> embed), followed by ReLU
        self.fc1_layers = nn.ModuleList()
        self.fc2_layers = nn.ModuleList()
        self.fc_final_layers = nn.ModuleList()
        for embed_dim in self.embed_dims:
            self.fc1_layers.append(nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(inplace=True),
            ))
            self.fc2_layers.append(nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(inplace=True),
            ))
            self.fc_final_layers.append(nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim),
                nn.ReLU(inplace=True),
            ))

        self.out_fc_layers = nn.ModuleList()
        for i in range(self.num_scales):
            out_fc = nn.Linear(self.embed_dims[i], multi_in_channels[i])
            self.out_fc_layers.append(nn.Sequential(
                out_fc,
                nn.ReLU(inplace=True),
            ))



    def forward(self, feat_list1, feat_list2):
        """
        feat_list1, feat_list2: Each is a list of feature tensors from image1 and image2.
                                Expected format: [feat_scale1, feat_scale2, feat_scale3, feat_scale4, context_feat],
                                where context_feat is the last high-level feature (to be broadcast).
        Returns: list of fused feature tensors for scales 1-4.
        """
        # Ensure the input lists have the expected number of features
        assert len(feat_list1) == len(feat_list2) == self.num_scales + 1, \
            "Feature lists must contain {} scale features + 1 context feature.".format(self.num_scales)

        # Split multi-scale features and context features
        context1 = feat_list1[-1]  # shape: B x C_ctx x 8 x 8 (context feature of image1)
        context2 = feat_list2[-1]  # shape: B x C_ctx x 8 x 8 (context feature of image2)
        multi_feats1 = feat_list1[:-1]  # list of 4 feature maps for image1
        multi_feats2 = feat_list2[:-1]  # list of 4 feature maps for image2

        fused_multi_feats1 = []  # to collect fused features for each scale
        fused_multi_feats2 = []


        # Iterate over each scale feature map.
        for i in range(self.num_scales):
            f1 = multi_feats1[i]  # feature map from image1 at scale i, shape: B x C_i x H_i x W_i
            f2 = multi_feats2[i]  # feature map from image2 at scale i, shape: B x C_i x H_i x W_i
            B, C_i, H_i, W_i = f1.size()

            # 1. Upsample/broadcast context features to match spatial size H_i x W_i
            c1_up = F.interpolate(context1, size=(H_i, W_i), mode='nearest')
            c2_up = F.interpolate(context2, size=(H_i, W_i), mode='nearest')
            # (Using nearest-neighbor upsampling to broadcast the context)

            # 2. Concatenate context in channel dimension
            # After concat: shape = B x (C_i + C_ctx) x H_i x W_i for each image
            f1_context = torch.cat([f1, c1_up], dim=1)
            f2_context = torch.cat([f2, c2_up], dim=1)
            C_embed = f1_context.size(1)  # this should equal self.embed_dims[i]

            # 3. Reshape to (B, N, C) where N = H_i * W_i (flatten spatial dimensions)
            # Permute so that each spatial location feature is a row in the sequence.
            f1_flat = f1_context.view(B, C_embed, H_i * W_i).permute(0, 2, 1)  # shape: B x N x C_embed
            f2_flat = f2_context.view(B, C_embed, H_i * W_i).permute(0, 2, 1)  # shape: B x N x C_embed

            # 4. Fuse features at this scale using concatenation and learned transforms (FusDiff style)
            # Compute element-wise product for paired features
            prod = f1_flat * f2_flat  # B x N x C_embed (elementwise multiplication)
            # Apply first linear layers to get transformed features for each image
            X_before = self.fc1_layers[i](torch.cat([prod, f1_flat], dim=-1))  # B x N x C_embed
            X_after = self.fc2_layers[i](torch.cat([prod, f2_flat], dim=-1))  # B x N x C_embed

            # Apply a linear layer on (B, N, C_embed), then reshape back to (B, C_i, H_i, W_i).
            X_before_out = self.out_fc_layers[i](X_before)  # B x N x C_i
            X_after_out = self.out_fc_layers[i](X_after)  # B x N x C_i

            # 5. Reshape fused features back to (B, C_embed, H_i, W_i)
            fused1 = X_before_out.permute(0, 2, 1).view(B, self.multi_in_channels[i], H_i, W_i)
            fused2 = X_after_out.permute(0, 2, 1).view(B, self.multi_in_channels[i], H_i, W_i)

            fused_multi_feats1.append(fused1)
            fused_multi_feats2.append(fused2)

        CC_img1_feat, CC_img2_feat = fused_multi_feats1[-1], fused_multi_feats2[-1]
        fused_multi_feats1.append(CC_img1_feat)
        fused_multi_feats2.append(CC_img2_feat)

        return fused_multi_feats1, fused_multi_feats2

# Critique module.
class TextAttentionPool(nn.Module):
    """
    Simple text attention pooling:
    Input: (L, B, D) -> output: (B, D)
    """
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))

    def forward(self, x):  # x: (L,B,D)
        L, B, D = x.shape
        q = self.query.view(1, 1, D).expand(L, B, D)
        score = (x * q).sum(-1)               # (L,B)
        attn  = F.softmax(score, dim=0)       # (L,B)
        out   = (attn.unsqueeze(-1) * x).sum(0)  # (B,D)
        return out


class MultimodalCritiqueModuleV3(nn.Module):
    def __init__(self,
                 visual_channels=128,    # Channel count of cd_feat.
                 text_dim=512,           # Text dimension.
                 hidden=128,             # Decoder hidden channels.
                 spatial_size=256,       # Output resolution, usually 2H=2W=256.
                 tau=1.0,                # Softmax temperature over visual tokens.
                 use_text_cond=True,     # Whether to use text-conditioned scoring.
                 use_groupnorm=False,    # True -> GroupNorm(32), False -> BatchNorm2d.
                 lambda_dice=0.5,        # Dice weight in forward().
                 lambda_bg=0.1):         # Background suppression weight in forward().
        super().__init__()
        self.text_dim      = text_dim
        self.spatial_size  = spatial_size
        self.tau           = tau
        self.use_text_cond = use_text_cond
        self.lambda_dice   = lambda_dice
        self.lambda_bg     = lambda_bg

        # Visual token -> text dimension.
        self.vis_proj = nn.Linear(visual_channels, text_dim, bias=False)

        # Text pooling and conditional parameters for additive attention.
        self.txt_pool = TextAttentionPool(text_dim)
        self.Wv = nn.Linear(text_dim, text_dim)
        self.Wt = nn.Linear(text_dim, text_dim)
        self.v  = nn.Linear(text_dim, 1)

        # Shared scoring layer when text conditioning is disabled, kept for compatibility.
        self.score_proj = nn.Linear(text_dim, 1)

        # Refine and upsample: HxW -> 2H x 2W.
        Norm2d = (lambda ch: nn.GroupNorm(32, ch)) if use_groupnorm else nn.BatchNorm2d

        self.decoder = nn.Sequential(
            nn.Conv2d(1, hidden, 3, 1, 1), Norm2d(hidden), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # Refinement layer; depthwise separable convolution can be used to reduce cost.
            nn.Conv2d(hidden, hidden // 2, 3, padding=1, bias=False),
            Norm2d(hidden // 2), nn.ReLU(inplace=True),

            nn.Conv2d(hidden // 2, 16, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()  # Remove this if switching later to BCEWithLogitsLoss.
        )
        # Bias the initial output toward background.
        nn.init.constant_(self.decoder[-2].bias, -2.0)   # Conv2d(16,1,1).bias
        nn.init.zeros_(self.decoder[-2].weight)

    # ---------- Internal: compute visual-token logits. ----------
    def _vis_logits(self, vis_seq, cc_seq, tau=None):
        """
        vis_seq: (N,B,D)  from cd_feat
        cc_seq : (L,B,D)  text tokens, teacher-forcing embedding + PE.
        return : vis_log (N,B,1)
        """
        tau = self.tau if tau is None else tau
        if self.use_text_cond:
            t = self.txt_pool(cc_seq)                                 # (B,D)
            t = t.unsqueeze(0).expand(vis_seq.size(0), -1, -1)       # (N,B,D)
            h = torch.tanh(self.Wv(vis_seq) + self.Wt(t))            # (N,B,D)
            vis_log = self.v(h)                                      # (N,B,1)
        else:
            vis_log = self.score_proj(vis_seq)                        # (N,B,1)
        return vis_log / tau

    # ---------- Adversarial interface: return weights a (B x N) and optional spatial mask. ----------
    def get_weights(self, cd_feat, cc_seq=None, tau=None,
                    detach_backbone=True, detach_text=True, return_mask=False):
        """
        Returns:
          a: (B, N=H*W), visual-token weights for each sample, normalized over N.
          critic_mask(optional): (B,1,2H,2W)
        Usage:
          - Train Critic: compute E with a and e128.detach(), then maximize E or minimize -E.
          - Train CD/CC: compute E with a.detach() and e128, then minimize E.
        """
        B, C, H, W = cd_feat.shape
        cd_in = cd_feat.detach() if detach_backbone else cd_feat
        vis_seq = cd_in.flatten(2).permute(2, 0, 1)                  # (N,B,C)
        vis_seq = self.vis_proj(vis_seq)                              # (N,B,D)

        if cc_seq is None:
            cc_in = torch.zeros(1, B, self.text_dim, device=cd_feat.device)
        else:
            cc_in = cc_seq.detach() if detach_text else cc_seq       # (L,B,D)

        vis_log = self._vis_logits(vis_seq, cc_in, tau=tau)          # (N,B,1)
        weights = F.softmax(vis_log, dim=0)                           # (N,B,1)

        a = weights.squeeze(-1).permute(1, 0).contiguous()           # (B,N)

        if not return_mask:
            return a

        crit128 = weights.permute(1, 2, 0).contiguous().view(B, 1, H, W)  # (B,1,H,W)
        critic_mask = self.decoder(crit128)                                # (B,1,2H,2W)
        return a, critic_mask

    # ---------- Critic objective: minimize -E plus regularization. ----------
    def critic_loss(self, e128, a, critic_mask=None, err_hard=None, valid_mask=None,
                    lambda_ent=1e-3, lambda_tv=0.0, lambda_align=0.0):
    
        Lc = (a * e128).sum(dim=1).mean()
        L_tgca = -Lc

        
        eps = 1e-8
        Ha = -(a * (a + eps).log()).sum(dim=1).mean()
        L_tgca = L_tgca + lambda_ent * Ha

       
        if critic_mask is not None and lambda_tv > 0.0:
            dy = critic_mask[:, :, 1:, :] - critic_mask[:, :, :-1, :]
            dx = critic_mask[:, :, :, 1:] - critic_mask[:, :, :, :-1]
            if valid_mask is not None:
                vm_y = valid_mask[:, :, 1:, :] * valid_mask[:, :, :-1, :]
                vm_x = valid_mask[:, :, :, 1:] * valid_mask[:, :, :, :-1]
                tv = (vm_y * dy.abs()).mean() + (vm_x * dx.abs()).mean()
            else:
                tv = dy.abs().mean() + dx.abs().mean()
            L_tgca = L_tgca + lambda_tv * tv

        if critic_mask is not None and err_hard is not None and lambda_align > 0.0:
            if valid_mask is None:
                valid_mask = torch.ones_like(err_hard)
            align = F.binary_cross_entropy(critic_mask, err_hard, reduction='none')
            align = (align * valid_mask).sum() / (valid_mask.sum() + 1e-6)
            def masked_mean(x): return (x * valid_mask).sum() / (valid_mask.sum() + 1e-6)
            loss_bg = masked_mean(critic_mask * (1. - err_hard))
            L_tgca = L_tgca + lambda_align * align + loss_bg

        return L_tgca, Lc

    
    def forward(self, cd_feat, cc_seq, label_mask, pred_mask,
                tau=None, detach_backbone_for_critic=True, detach_text_for_critic=True):
        
        B, C, H, W = cd_feat.shape
        device = cd_feat.device
        tau = self.tau if tau is None else tau

        # Visual features -> tokens.
        cd_in = cd_feat.detach() if detach_backbone_for_critic else cd_feat
        vis_seq = self.vis_proj(cd_in.flatten(2).permute(2, 0, 1))   # (N,B,D)

        # Text sequence.
        cc_in = cc_seq.detach() if detach_text_for_critic else cc_seq

        # Token logits -> weights -> spatial mask.
        vis_log = self._vis_logits(vis_seq, cc_in, tau=tau)          # (N,B,1)
        weights = F.softmax(vis_log, dim=0)                           # (N,B,1)
        crit128 = weights.permute(1, 2, 0).contiguous().view(B, 1, H, W)
        critic_mask = self.decoder(crit128)                           # (B,1,2H,2W)

        # Soft/hard error maps with ignore=2.
        pred_prob = torch.softmax(pred_mask, 1)[:, 1:].sum(1, keepdim=True)  # (B,1,256,256)
        label_bin = (label_mask == 1).float().unsqueeze(1)                    # (B,1,256,256)
        ignore_m  = (label_mask == 2).float().unsqueeze(1)
        valid_m   = 1.0 - ignore_m

        err_soft  = (label_bin - pred_prob).abs() * valid_m
        hard      = (pred_prob > 0.5).float()
        err_hard  = (label_bin != hard).float() * valid_m
        err_hard  = F.max_pool2d(err_hard, 3, 1, 1)

        if critic_mask.shape[2:] != err_soft.shape[2:]:
            critic_mask = F.interpolate(critic_mask, err_soft.shape[2:], mode='bilinear', align_corners=False)

        def masked_mean(x): return (x * valid_m).sum() / (valid_m.sum() + 1e-6)

        loss_soft = masked_mean((critic_mask - err_soft).abs())
        inter = (critic_mask * err_hard).sum()
        union = (critic_mask + err_hard).sum()
        loss_dice = 1 - (2 * inter + 1e-6) / (union + 1e-6)
        loss_bg = masked_mean(critic_mask * (1. - err_hard))

        loss_align = loss_soft + self.lambda_dice * loss_dice + self.lambda_bg * loss_bg
        return critic_mask, loss_align






