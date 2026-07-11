import os
import torch
import numpy as np
from eval_func.bleu.bleu import Bleu
from eval_func.rouge.rouge import Rouge
from eval_func.cider.cider import Cider
from eval_func.meteor.meteor import Meteor
import time
import torch.nn.functional as F
import random
import torch, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import torchvision.utils as vutils

def save_checkpoint(args, data_name, epoch, encoder, encoder_feat, decoder, encoder_optimizer,
                encoder_feat_optimizer, decoder_optimizer, best_bleu4):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'best_bleu-4': best_bleu4,
             'encoder': encoder,
             'encoder_feat': encoder_feat,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'encoder_feat_optimizer': encoder_feat_optimizer,
             'decoder_optimizer': decoder_optimizer,
             }
    #filename = 'checkpoint_' + data_name + '_' + args.network + '.pth.tar'
    path = args.savepath #'./models_checkpoint/mymodel/3-times/'
    if os.path.exists(path)==False:
        os.makedirs(path)
        # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    torch.save(state, os.path.join(path, 'BEST_' + data_name))

    # torch.save(state, os.path.join(path, 'checkpoint_' + data_name +'_epoch_'+str(epoch) + '.pth.tar'))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def get_eval_score(references, hypotheses):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    hypo = [[' '.join(hypo)] for hypo in [[str(x) for x in hypo] for hypo in hypotheses]]
    ref = [[' '.join(reft) for reft in reftmp] for reftmp in
           [[[str(x) for x in reft] for reft in reftmp] for reftmp in references]]
    score = []
    method = []
    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(ref, hypo)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
        #print("{} {}".format(method_i, score_i))
    score_dict = dict(zip(method, score))

    return score_dict


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
                
def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def time_file_str():
    ISOTIMEFORMAT='%Y-%m-%d-%H-%M-%S'
    string = '{}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
    return string #+ '-{}'.format(random.randint(1, 10000))

def print_log(print_string, log):
    print("{:}".format(print_string))
    log.write('{:}\n'.format(print_string))
    log.flush()

def freeze_module(mod: torch.nn.Module) -> None:
    """将一个 nn.Module 的全部参数 requires_grad 设为 False。"""
    mod.eval()                         # 关闭 BN/Dropout 的训练态（可选）
    for p in mod.parameters():
        p.requires_grad = False


def unfreeze_module(mod: torch.nn.Module) -> None:
    """恢复梯度；保持 .train()，以便 BN 能更新统计量。"""
    mod.train()
    for p in mod.parameters():
        p.requires_grad = True

def safe_autograd_grad(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]

def compute_pcgrad(g1_list, g2_list):
    g1 = torch.cat([g.reshape(-1) for g in g1_list])
    g2 = torch.cat([g.reshape(-1) for g in g2_list])
    dot = torch.dot(g1, g2)
    if dot < 0:
        proj = dot / (g2.norm() ** 2 + 1e-6)
        g1 = g1 - proj * g2
    return g1

def split_tensor_like(tensor, params):
    chunks = []
    offset = 0
    for p in params:
        numel = p.numel()
        chunk = tensor[offset:offset + numel].view_as(p)
        chunks.append(chunk)
        offset += numel
    return chunks

def compute_adv_loss1(pred_seg: torch.Tensor,
                         critic_mask: torch.Tensor,
                         eps: float = 1e-6):
    """
    计算分割分支与批评分支的对抗 IoU loss。

    Args:
        pred_seg:    [B, C, H, W] — CD 分支网络输出
        critic_mask: [B, C, H, W] — 批评分支输出的连续值 mask (0~1)
        eps:         防止除零的小量

    Returns:
        loss_cd_adv:     标量，分割分支最小化 IoU
        loss_critic_adv: 标量，批评分支最小化 -IoU
        iou_per_class:   [B, C] 每个样本每个通道的 IoU 值
    """
    B, C, H, W = pred_seg.shape

    # 1) 将 pred_seg → discrete labels → one-hot
    pred_labels = torch.argmax(pred_seg, dim=1)  # [B, H, W]
    seg_onehot = F.one_hot(pred_labels, num_classes=C) \
        .permute(0, 3, 1, 2) \
        .float()  # [B, C, H, W]

    # 2) Intersection & Union
    intersection = (seg_onehot * critic_mask).sum(dim=(2, 3))  # [B, C]
    union = (seg_onehot + critic_mask - seg_onehot * critic_mask).sum(dim=(2, 3))  # [B, C]

    # 3) IoU per class
    iou_per_class = (intersection + eps) / (union + eps)  # [B, C]

    # 4) 对所有样本、所有通道求平均
    iou_mean = iou_per_class.mean()  # scalar
    loss_adv = iou_mean  # CD 分支希望最小化 IoU

    return loss_adv

def compute_adv_loss(seg_pre: torch.Tensor,
                     critic_mask: torch.Tensor,
                     eps: float = 1e-6,
                     change_only: bool = True):
    """Adversarial IoU loss.

    * seg_pre: logits or softmax, [B,3,H,W]
    * critic_mask: sigmoid mask, [B,3 or 1,H,W]
    * change_only: if True, only class‑1 & class‑2 participate in IoU.
    """
    # --- Predictions → one‑hot ---
    with torch.no_grad():
        labels = torch.argmax(seg_pre, dim=1)              # [B,H,W]
    seg_onehot = F.one_hot(labels, num_classes=3).permute(0,3,1,2).float()

    # --- align critic_mask spatial size ---
    if seg_onehot.shape[2:] != critic_mask.shape[2:]:
        critic_mask = F.interpolate(critic_mask, size=seg_onehot.shape[2:], mode='bilinear', align_corners=False)

    # --- ensure 3‑channels ---
    if critic_mask.shape[1] == 1:
        critic_mask = critic_mask.repeat(1, 3, 1, 1)

    if change_only:
        seg_onehot = seg_onehot[:, 1:, :, :]     # class‑1 & 2
        critic_mask = critic_mask[:, 1:, :, :]

    inter = (seg_onehot * critic_mask).sum(dim=(2,3))
    union = (seg_onehot + critic_mask - seg_onehot * critic_mask).sum(dim=(2,3))
    iou   = (inter + eps) / (union + eps)        # [B, C_change]
    return 1 - iou.mean()                        # scalar loss



# --------- 颜色映射辅助 ---------
def _label_rgb(lbl: np.ndarray):
    """0/1/2 → 黑 / 红 / 黄"""
    palette = np.array([[0,0,0], [255,0,0], [255,255,0]], dtype=np.uint8)
    return palette[lbl.clip(0,2)]

def _heatmap(x: np.ndarray, cmap='inferno'):
    """0~1 / prob → heatmap RGB"""
    import matplotlib.cm as cm
    m = cm.get_cmap(cmap)
    return (m(x.clip(0,1))[:,:,:3]*255).astype(np.uint8)

# --------- 可视化函数 ---------
def save_mask_pairs(label_masks : torch.Tensor,
                    pred_masks  : torch.Tensor,
                    critic_masks: torch.Tensor,
                    out_dir='critic_vis',
                    max_show=4,
                    prefix='pair',
                    show_error=True):

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    label_np  = label_masks[:max_show].detach().cpu().numpy()          # ← detach
    # ---------- pred ----------
    if pred_masks.dim() == 4 and pred_masks.size(1) == 3:  # 三通道 logits
        pred_np = pred_masks[:max_show].detach().cpu().argmax(1).numpy()  # 0/1/2
        pred_rgb = [_label_rgb(p) for p in pred_np]
        # >>> 这里变 Torch，保持 bool → float
        pred_bin = torch.from_numpy((pred_np > 0).astype(np.float32)) \
            .unsqueeze(1)  # (B,1,H,W)
    else:  # 单通道 prob
        prob = pred_masks[:max_show, 0].detach().cpu()  # (B,H,W)
        pred_np = (prob > 0.5).numpy().astype(np.uint8)
        pred_rgb = [_heatmap(prob[i].numpy()) for i in range(prob.size(0))]
        pred_bin = (prob > 0.5).float().unsqueeze(1)  # (B,1,H,W)
    # ---------- 批评 ----------
    if critic_masks.dim()==4 and critic_masks.size(1)==3:
        critic_np  = critic_masks[:max_show].detach().cpu().numpy().transpose(0,2,3,1)
        critic_rgb = [(c*255).astype(np.uint8) for c in critic_np]
    else:
        critic_np  = critic_masks[:max_show,0].detach().cpu().numpy()  # ← detach
        critic_rgb = [_heatmap(c) for c in critic_np]

    # ---------- error-map (硬 XOR + 3×3 膨胀) ----------
    if show_error:
        lbl_tensor = (torch.from_numpy(label_np) > 0).float().unsqueeze(1)  # (B,1,H,W)
        err_hard = (lbl_tensor != pred_bin).float()
        err_hard = F.max_pool2d(err_hard, 3, stride=1, padding=1)  # ★同训练
        err_rgb = [_heatmap(e.squeeze().numpy()) for e in err_hard]  # list[(H,W,3)]
    saved = []
    for i in range(len(label_np)):
        ncols = 4 if show_error else 3
        fig, ax = plt.subplots(1, ncols, figsize=(3 * ncols, 3))

        ax[0].imshow(_label_rgb(label_np[i]));
        ax[0].set_title('label')
        ax[1].imshow(pred_rgb[i]);
        ax[1].set_title('pred')
        if show_error:
            ax[2].imshow(err_rgb[i]);
            ax[2].set_title('error')
            ax[3].imshow(critic_rgb[i]);
            ax[3].set_title('critic')
        else:
            ax[2].imshow(critic_rgb[i]);
            ax[2].set_title('critic')
        for a in ax: a.axis('off')

        f = out_dir / f'{prefix}_{i}.png'
        plt.tight_layout();
        fig.savefig(f, dpi=150);
        plt.close(fig)
        saved.append(f.name)

    print(f'[save_mask_pairs] saved: {saved} → {out_dir.resolve()}')


def iou_loss(pred_mask, critic_mask, eps=1e-6):
    inter = (pred_mask * critic_mask).sum((1,2,3))
    union = (pred_mask + critic_mask - pred_mask*critic_mask).sum((1,2,3))
    return 1. - (inter + eps) / (union + eps)

def critic_region_bce(pred_prob, label_bin, critic_mask, eps=1e-7):
    """
    BCE 仅在批评区域计算，自动把输入 clamp 到 (eps, 1-eps)
    """
    inp = (pred_prob * critic_mask).clamp(eps, 1. - eps)   # ← 关键
    tgt = (label_bin * critic_mask)                        # 0 / 1
    return F.binary_cross_entropy(inp, tgt)

def critic_contrastive_loss(critic_mask: torch.Tensor,
                            error_map  : torch.Tensor,
                            tau: float = 0.1) -> torch.Tensor:
    """
    critic_mask : (B,1,H,W) – Critic-Net Sigmoid 输出
    error_map   : (B,1,H,W) – (label_bin != pred_bin) 得到的 0/1 错误图
    """
    B = critic_mask.size(0)
    c = F.normalize(critic_mask.flatten(1), dim=1)
    e = F.normalize(error_map  .flatten(1), dim=1)
    sim    = torch.mm(c, e.t())          # (B,B) 余弦相似
    logits = sim / tau
    labels = torch.arange(B, device=critic_mask.device)
    return F.cross_entropy(logits, labels)

# ---- 统一配色：背景=黑，类1=黄(1,1,0)，类2=红(1,0,0) ----
PALETTE = {0:(0.,0.,0.), 1:(1.,1.,0.), 2:(1.,0.,0.)}  # 黑/黄/红

def colorize_discrete(mask_hw: torch.Tensor) -> torch.Tensor:
    h, w = mask_hw.shape
    rgb = torch.zeros(3, h, w, dtype=torch.float32, device=mask_hw.device)
    for k,(r,g,b) in PALETTE.items():
        m = (mask_hw == k)
        if m.any():
            rgb[0][m]=r; rgb[1][m]=g; rgb[2][m]=b
    return rgb

def _norm01(x: torch.Tensor, mode="minmax", eps=1e-6) -> torch.Tensor:
    # x: (1,H,W) or (H,W)
    if x.dim()==2: x = x.unsqueeze(0)
    x = x.float()
    if mode == "minmax":
        mn = torch.amin(x, dim=(1,2), keepdim=True)
        mx = torch.amax(x, dim=(1,2), keepdim=True)
        x = (x - mn) / (mx - mn + eps)
    elif mode == "p2p98":
        # 更鲁棒：2%-98%分位
        flat = x.flatten(1)
        lo = torch.quantile(flat, 0.02, dim=1, keepdim=True).unsqueeze(-1)
        hi = torch.quantile(flat, 0.98, dim=1, keepdim=True).unsqueeze(-1)
        x = (x - lo) / (hi - lo + eps)
    return x.clamp(0,1)

def colorize_heat_yellow(x01: torch.Tensor) -> torch.Tensor:
    # x01: (1,H,W) in [0,1]
    x01 = x01.clamp(0,1)
    return torch.cat([x01, x01, torch.zeros_like(x01)], dim=0)  # (3,H,W) 黄=R+G

def save_viz4_unified(save_dir, tag,
                      seg_pre: torch.Tensor,      # (B,C,H,W) logits
                      label_mask: torch.Tensor,   # (B,H,W) long
                      critic_mask: torch.Tensor,  # (B,1,H,W) float
                      err_hard: torch.Tensor,     # (B,1,H,W) float/0-1
                      max_n=4, norm_mode="p2p98") -> str:
    os.makedirs(save_dir, exist_ok=True)
    B, _, H, W = seg_pre.shape
    n = min(B, max_n)

    with torch.no_grad():
        pred_cls = seg_pre.argmax(dim=1)  # (B,H,W)
        rows = []
        for i in range(n):
            pred_rgb = colorize_discrete(pred_cls[i].cpu())
            gt_rgb   = colorize_discrete(label_mask[i].cpu().long())

            cm = critic_mask[i].cpu()
            eh = err_hard[i].cpu()
            # ★ 关键：先归一化到0-1再上色，避免“整块黑”
            cm_rgb = colorize_heat_yellow(_norm01(cm, mode=norm_mode))
            eh_rgb = colorize_heat_yellow(_norm01(eh, mode=norm_mode))

            row = torch.stack([pred_rgb, gt_rgb, cm_rgb, eh_rgb], dim=0)  # (4,3,H,W)
            rows.append(row)

        grid = vutils.make_grid(torch.cat(rows, dim=0), nrow=4, padding=2)
        out_path = os.path.join(save_dir, f'{tag}.png')
        vutils.save_image(grid, out_path)
        return out_path


