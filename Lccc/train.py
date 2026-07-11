import torch.optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils import data
import argparse
import json
from tqdm import tqdm
from data.LEVIR_MCI import LEVIRCCDataset
from data.WHU_CDC import WHUCDCDataset
from model.model_encoder_att import Encoder, AttentiveEncoder1
from model.model_decoder import DecoderTransformerV1
from utils_tool.utils import *
from utils_tool.metrics import Evaluator
from model.model_t import MultiScaleAttentionSuppression, MultiScaleFeatureFusion, MultimodalCritiqueModuleV3
import torch.nn.functional as F



class Trainer(object):
    def __init__(self, args):
        """
        Training and validation.
        """
        self.start_train_goal = args.train_goal
        self.args = args
        torch.cuda.set_device(args.gpu_id)
        random_str = str(random.randint(10, 100))
        name = 'baseline_'+time_file_str() + f'_train_goal_{args.train_goal}_' + random_str
        self.args.savepath = os.path.join(args.savepath, name)
        self.args.savepath = os.path.join(args.savepath, name)
        if os.path.exists(self.args.savepath)==False:
            os.makedirs(self.args.savepath)
        self.log = open(os.path.join(self.args.savepath, '{}.log'.format(name)), 'w')
        print_log('=>datset: {}'.format(args.data_name), self.log)
        print_log('=>network: {}'.format(args.network), self.log)
        print_log('=>encoder_lr: {}'.format(args.encoder_lr), self.log)
        print_log('=>decoder_lr: {}'.format(args.decoder_lr), self.log)
        print_log('=>num_epochs: {}'.format(args.num_epochs), self.log)
        print_log('=>train_batchsize: {}'.format(args.train_batchsize), self.log)

        self.best_bleu4 = 0.4  # BLEU-4 score right now
        self.MIou = 0.4
        self.Sum_Metric = 0.4
        self.start_epoch = 0
        with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
            self.word_vocab = json.load(f)
        # Initialize / load checkpoint
        self.build_model()

        # Loss function
        self.criterion_cap = torch.nn.CrossEntropyLoss().cuda()
        self.criterion_det = torch.nn.CrossEntropyLoss().cuda()

        # Custom dataloaders
        if args.data_name == 'LEVIR_MCI':
            self.train_loader = data.DataLoader(
                LEVIRCCDataset(args.data_folder, args.list_path, 'train', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
                batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
            self.val_loader = data.DataLoader(
                LEVIRCCDataset(args.data_folder, args.list_path, 'val', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
                batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
        elif args.data_name == 'WHU_CDC':
            self.train_loader = data.DataLoader(
                WHUCDCDataset(args.data_folder, args.list_path, 'train', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
                batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
            self.val_loader = data.DataLoader(
                WHUCDCDataset(args.data_folder, args.list_path, 'val', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
                batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
        else:
            raise ValueError(f'Unsupported data_name: {args.data_name}')

        self.index_i = 0
        self.hist = np.zeros((args.num_epochs*2 * len(self.train_loader), 5))
        # Epochs

        self.evaluator = Evaluator(num_class=3)

        self.best_model_path = None
        self.best_epoch = 0

    def build_model(self):
        args = self.args

        # ---------- Stage 1: build from scratch ----------
        if args.train_stage == 's1':
            self.encoder = Encoder(args.network)
            self.encoder.fine_tune(args.fine_tune_encoder)

            self.encoder_trans = AttentiveEncoder1(
                train_stage=args.train_stage, n_layers=args.n_layers,
                feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
                heads=args.n_heads, dropout=args.dropout
            )

            self.decoder = DecoderTransformerV1(
                encoder_dim=args.encoder_dim, feature_dim=args.feature_dim,
                vocab_size=len(self.word_vocab), max_lengths=args.max_length,
                word_vocab=self.word_vocab, n_head=args.n_heads,
                n_layers=args.decoder_n_layers, dropout=args.dropout
            )

            # Critic
            self.critic = MultimodalCritiqueModuleV3(
                visual_channels=128, text_dim=args.feature_dim, hidden=128,
                spatial_size=256, tau=args.tau,
                use_text_cond=True, use_groupnorm=False,
                lambda_dice=0.5, lambda_bg=0.1
            )

            # RSE and KCFF
            self.Suppression_module = MultiScaleAttentionSuppression(
                [64, 128, 320, 512], 512, 256, True, 0.1
            )
            self.Complementary_module = MultiScaleFeatureFusion()

            fine_tune_capdecoder = True

        # ---------- Stage 2: load weights and switch by goal ----------
        elif args.train_stage == 's2' and args.checkpoint is not None:
            checkpoint = torch.load(args.checkpoint, map_location='cuda')
            print('Load Model from {}'.format(args.checkpoint))

            # Ensure modules exist 
            if not hasattr(self, 'encoder'):
                self.encoder = Encoder(args.network)
            if not hasattr(self, 'encoder_trans'):
                self.encoder_trans = AttentiveEncoder1(
                    train_stage=args.train_stage, n_layers=args.n_layers,
                    feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
                    heads=args.n_heads, dropout=args.dropout
                )
            if not hasattr(self, 'decoder'):
                self.decoder = DecoderTransformerV1(
                    encoder_dim=args.encoder_dim, feature_dim=args.feature_dim,
                    vocab_size=len(self.word_vocab), max_lengths=args.max_length,
                    word_vocab=self.word_vocab, n_head=args.n_heads,
                    n_layers=args.decoder_n_layers, dropout=args.dropout
                )
            if not hasattr(self, 'critic'):
                self.critic = MultimodalCritiqueModuleV3(
                    visual_channels=128, text_dim=args.feature_dim, hidden=128,
                    spatial_size=256, tau=args.tau,
                    use_text_cond=True, use_groupnorm=False,
                    lambda_dice=0.5, lambda_bg=0.1
                )
            if not hasattr(self, 'Suppression_module'):
                self.Suppression_module = MultiScaleAttentionSuppression(
                    [64, 128, 320, 512], 512, 256, True, 0.1
                )
            if not hasattr(self, 'Complementary_module'):
                self.Complementary_module = MultiScaleFeatureFusion()

            # Load main network weights.
            self.decoder.load_state_dict(checkpoint['decoder_dict'])
            self.encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'], strict=False)
            self.encoder.load_state_dict(checkpoint['encoder_dict'])

            # Optionally load critic and auxiliary module weights.
            if 'critic_dict' in checkpoint:
                try:
                    self.critic.load_state_dict(checkpoint['critic_dict'], strict=False)
                except Exception as e:
                    print(f'Warn: failed to load critic_dict: {e}')
            if 'suppression_dict' in checkpoint:
                try:
                    self.Suppression_module.load_state_dict(checkpoint['suppression_dict'], strict=False)
                except Exception as e:
                    print(f'Warn: failed to load suppression_dict: {e}')
            if 'complementary_dict' in checkpoint:
                try:
                    self.Complementary_module.load_state_dict(checkpoint['complementary_dict'], strict=False)
                except Exception as e:
                    print(f'Warn: failed to load complementary_dict: {e}')

            # Stage 2 fine-tuning strategy.
            args.fine_tune_encoder = False
            self.encoder.fine_tune(False)
            self.encoder_trans.fine_tune(args.train_goal)

            # Freeze decoder for goal 0; train it for goals 1 and 2.
            fine_tune_capdecoder = (args.train_goal != 0)
            if hasattr(self.decoder, 'fine_tune'):
                if args.train_goal == 1:
                    self.decoder.fine_tune(True, freeze_embeddings=True, unfreeze_last_n=1, verbose=True)
                else:
                    self.decoder.fine_tune(fine_tune_capdecoder)
            else:
                for p in self.decoder.parameters():
                    p.requires_grad = fine_tune_capdecoder
        else:
            raise ValueError('Error: checkpoint is None.')

        # ---------- Move modules to GPU ----------
        self.encoder = self.encoder.cuda()
        self.encoder_trans = self.encoder_trans.cuda()
        self.decoder = self.decoder.cuda()
        if self.critic is not None: self.critic = self.critic.cuda()
        self.Suppression_module = self.Suppression_module.cuda()
        self.Complementary_module = self.Complementary_module.cuda()

        # ---------- Optimizers ----------
        def _maybe_adam(params, lr):
            params = [p for p in params if p.requires_grad]
            return torch.optim.Adam(params, lr=lr) if len(params) > 0 else None

        # Train the encoder only in stage 1 when fine_tune_encoder is enabled.
        self.encoder_optimizer = _maybe_adam(self.encoder.parameters(),
                                             args.encoder_lr) if args.fine_tune_encoder else None
        # Optimize only encoder_trans parameters here.
        self.encoder_trans_optimizer = _maybe_adam(self.encoder_trans.parameters(), args.encoder_lr)
        # decoder
        self.decoder_optimizer = _maybe_adam(self.decoder.parameters(),
                                             args.decoder_lr) if fine_tune_capdecoder else None
        # Use the critic only for goals 0 and 2.
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr) \
            if (self.critic is not None and args.train_goal in (0, 2)) else None
        # Use independent fixed-lr optimizers for suppression and complementary modules.
        sup_lr, comp_lr = 1e-5, 1e-5
        self.suppression_optimizer = _maybe_adam(self.Suppression_module.parameters(), sup_lr)
        self.complementary_optimizer = _maybe_adam(self.Complementary_module.parameters(), comp_lr)

        # ---------- Schedulers ----------
        self.encoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_optimizer, step_size=5,
                                                                    gamma=1.0) if self.encoder_optimizer else None
        self.encoder_trans_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_trans_optimizer, step_size=5,
                                                                          gamma=1.0) if self.encoder_trans_optimizer else None
        self.decoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.decoder_optimizer, step_size=5,
                                                                    gamma=1.0) if self.decoder_optimizer else None
        self.critic_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=5,
                                                                   gamma=1.0) if self.critic_optimizer else None
        self.suppression_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.suppression_optimizer, step_size=5,
                                                                        gamma=1.0) if self.suppression_optimizer else None
        self.complementary_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.complementary_optimizer, step_size=5,
                                                                          gamma=1.0) if self.complementary_optimizer else None

    def training(self, args, epoch):

        # ------- Switch train/eval modes -------
        if self.args.train_goal == 2:
            self.encoder.train()
            self.encoder_trans.train()
            self.decoder.train()
            if self.critic is not None: self.critic.train()
            self.Suppression_module.train()
            self.Complementary_module.train()
        elif self.args.train_goal == 1:
            self.encoder.eval()
            self.encoder_trans.eval()
            self.decoder.train()
            if self.critic is not None: self.critic.eval()
            self.Suppression_module.eval()
            self.Complementary_module.eval()
        else:  # goal==0
            self.encoder.eval()
            self.encoder_trans.train()
            self.decoder.eval()
            if self.critic is not None: self.critic.train()
            self.Suppression_module.train()
            self.Complementary_module.train()

        # ------- Clear gradients -------
        if self.decoder_optimizer is not None: self.decoder_optimizer.zero_grad()
        if self.encoder_trans_optimizer is not None: self.encoder_trans_optimizer.zero_grad()
        if self.encoder_optimizer is not None: self.encoder_optimizer.zero_grad()
        if self.critic_optimizer is not None: self.critic_optimizer.zero_grad()
        if self.suppression_optimizer is not None: self.suppression_optimizer.zero_grad()
        if self.complementary_optimizer is not None: self.complementary_optimizer.zero_grad()

        for id, (imgA, imgB, seg_label, _, _, token, token_len, _) in enumerate(self.train_loader):
            start_time = time.time()
            accum_steps = max(1, 64 // args.train_batchsize)

            # ------- Move tensors to device -------
            imgA = imgA.cuda()
            imgB = imgB.cuda()
            seg_label = seg_label.cuda()
            token = token.squeeze(1).cuda()
            token_len = token_len.cuda()

            # ======= Forward pass =======
            if self.args.train_goal == 1:
                # Fine-tune only the captioning branch; keep upstream modules in no_grad.
                with torch.no_grad():
                    feat1, feat2 = self.encoder(imgA, imgB)
                    feat1, feat2 = self.Suppression_module(feat1, feat2)  # NEW
                    feat1, feat2 = self.Complementary_module(feat1, feat2)  # NEW
                    feat1, feat2, seg_pre, _ = self.encoder_trans(feat1, feat2)
            else:
                feat1, feat2 = self.encoder(imgA, imgB)
                feat1, feat2 = self.Suppression_module(feat1, feat2)  # NEW
                feat1, feat2 = self.Complementary_module(feat1, feat2)  # NEW
                feat1, feat2, seg_pre, cd_pred_features = self.encoder_trans(feat1, feat2)

            # Captioning forward pass when goal is not detection-only.
            cap_loss = None
            if self.args.train_goal != 0:
                try:
                    scores, caps_sorted, decode_lengths, sort_ind = self.decoder(feat1, feat2, token, token_len,
                                                                                 seg_pre=seg_pre)
                except TypeError:
                    scores, caps_sorted, decode_lengths, sort_ind = self.decoder(feat1, feat2, token, token_len)
                targets = caps_sorted[:, 1:]
                scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
                cap_loss = self.criterion_cap(scores, targets.long())

            # Change detection loss.
            det_loss = self.criterion_det(seg_pre, seg_label.long())

            # ======= Adversarial block for goals 0 and 2 =======
            E = seg_pre.new_zeros(())
            if (self.critic is not None) and (self.args.train_goal in (0, 2)):
                # Detached text condition for the critic.
                cc_seq_for_critic = self.decoder.position_encoding(
                    self.decoder.vocab_embedding(token).transpose(1, 0)
                ).detach()
                # cc_seq_for_critic = self.decoder.get_last_hidden_from_inputs(
                #     x1, x2, token,
                #     seg_pre=seg_pre,
                #     is_logits=True,
                #     token_mode="per_class",
                #     change_idx=(1, 2),
                #     detach_mask=True,
                #     gamma=1.0,
                #     detach=True,
                #     build_pad_from_vocab=True,
                #     mem_append_tokens=True
                # )  # (L, B, D)

                # Error map.
                # prob = torch.softmax(seg_pre, dim=1)
                # p_fg = prob[:, 1:].sum(1, keepdim=True)
                # y_bin = (seg_label == 1).float().unsqueeze(1)
                # valid = (seg_label != 2).float().unsqueeze(1)
                #
                # err_soft = (p_fg - y_bin).abs() * valid
                # hard = (p_fg > 0.5).float()
                # err_hard = (hard != y_bin).float() * valid
                # err_hard = F.max_pool2d(err_hard, 3, 1, 1)
                prob = torch.softmax(seg_pre, dim=1)  # (B,3,H,W)
                p_fg = prob[:, 1:].sum(1, keepdim=True)  # Foreground probability: class 1 + class 2.

                y_bin = (seg_label > 0).float().unsqueeze(1)  # GT foreground: classes 1 and 2 are foreground.
                # If there is no ignore class:
                valid = torch.ones_like(y_bin)  # All pixels are valid.
                # If the dataset has an ignore ID such as 255, use:
                # valid = (seg_label != ignore_id).float().unsqueeze(1)

                err_soft = (p_fg - y_bin).abs() * valid
                hard = (p_fg > 0.5).float()
                err_hard = (hard != y_bin).float() * valid
                err_hard = F.max_pool2d(err_hard, 3, 1, 1)

                # Downsample to the critic token space.
                Ht = 128
                e128 = F.adaptive_avg_pool2d(err_soft, (Ht, Ht)).flatten(1)
                cd_feat = F.adaptive_avg_pool2d(cd_pred_features, (Ht, Ht))

                # Critic step: maximize
                for _ in range(args.n_critic):
                    a, critic_mask = self.critic.get_weights(cd_feat, cc_seq_for_critic, return_mask=True)
                    Lc, E_now = self.critic.critic_loss(
                        e128.detach(), a,
                        critic_mask=critic_mask,
                        err_hard=err_hard.detach(),
                        valid_mask=valid,
                        lambda_ent=args.lambda_ent,
                        lambda_tv=args.lambda_tv,
                        lambda_align=args.lambda_align
                    )
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    Lc.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), args.critic_grad_clip)
                    self.critic_optimizer.step()
                    if self.critic_lr_scheduler is not None:
                        self.critic_lr_scheduler.step()

                # Gener Correction step
                with torch.no_grad():
                    a = self.critic.get_weights(cd_feat.detach(), cc_seq_for_critic)
                E = (a * e128).sum(dim=1).mean()
                lambda_adv = args.lambda_adv * min(1.0, epoch / max(1, args.adv_warm_epochs))

            else:
                lambda_adv = 0.0

            # ======= Compose loss =======
            if self.args.train_goal == 0:
                loss = det_loss + lambda_adv * E
            elif self.args.train_goal == 1:
                loss = cap_loss
            else:  # goal==2
                if args.train_stage == 's1':
                    det_loss = det_loss / det_loss.detach().item()
                    cap_loss = cap_loss / cap_loss.detach().item()
                loss = det_loss + cap_loss + lambda_adv * E

            # ------- Backpropagation / accumulation -------
            loss = loss / accum_steps
            loss.backward()

            # Clip gradients for modules that are active.
            if args.grad_clip is not None:
                if self.decoder_optimizer is not None:
                    torch.nn.utils.clip_grad_value_(self.decoder.parameters(), args.grad_clip)
                if self.encoder_trans_optimizer is not None:
                    torch.nn.utils.clip_grad_value_(self.encoder_trans.parameters(), args.grad_clip)
                if self.encoder_optimizer is not None:
                    torch.nn.utils.clip_grad_value_(self.encoder.parameters(), args.grad_clip)
                if self.suppression_optimizer is not None:
                    torch.nn.utils.clip_grad_value_(self.Suppression_module.parameters(), args.grad_clip)
                if self.complementary_optimizer is not None:
                    torch.nn.utils.clip_grad_value_(self.Complementary_module.parameters(), args.grad_clip)

            # ------- Optimizer step -------
            if (id + 1) % accum_steps == 0 or (id + 1) == len(self.train_loader):
                if self.decoder_optimizer is not None:
                    self.decoder_optimizer.step()
                    self.decoder_optimizer.zero_grad()
                if self.encoder_trans_optimizer is not None:
                    self.encoder_trans_optimizer.step()
                    self.encoder_trans_optimizer.zero_grad()
                if self.encoder_optimizer is not None:
                    self.encoder_optimizer.step()
                    self.encoder_optimizer.zero_grad()
                # These modules receive gradients only for goals 0 and 2.
                if self.suppression_optimizer is not None and self.args.train_goal in (0, 2):
                    self.suppression_optimizer.step()
                    self.suppression_optimizer.zero_grad()
                if self.complementary_optimizer is not None and self.args.train_goal in (0, 2):
                    self.complementary_optimizer.step()
                    self.complementary_optimizer.zero_grad()

                # Step schedulers.
                if self.decoder_lr_scheduler is not None: self.decoder_lr_scheduler.step()
                if self.encoder_trans_lr_scheduler is not None: self.encoder_trans_lr_scheduler.step()
                if self.encoder_lr_scheduler is not None: self.encoder_lr_scheduler.step()
                if self.suppression_lr_scheduler is not None and self.args.train_goal in (0, 2):
                    self.suppression_lr_scheduler.step()
                if self.complementary_lr_scheduler is not None and self.args.train_goal in (0, 2):
                    self.complementary_lr_scheduler.step()

            # ------- Logging -------
            self.hist[self.index_i, 0] = time.time() - start_time
            if self.args.train_goal in (0, 2):
                self.hist[self.index_i, 1] = det_loss.item()
                self.hist[self.index_i, 2] = accuracy(
                    seg_pre.permute(0, 2, 3, 1).reshape(-1, seg_pre.size(1)),
                    seg_label.reshape(-1), 1
                )
            if self.args.train_goal in (1, 2):
                self.hist[self.index_i, 3] = cap_loss.item()
                self.hist[self.index_i, 4] = accuracy(scores, targets, 5)

            self.index_i += 1

            if self.index_i % args.print_freq == 0:
                print_log(
                    'Training Epoch: [{0}][{1}/{2}]  '
                    'Batch Time: {3:.3f}  '
                    'Det_Loss: {4:.4f}  Det Acc: {5:.3f}  '
                    'Cap_Loss: {6:.5f}  Text_Top-5: {7:.3f}'.format(
                        epoch, id, len(self.train_loader),
                        np.mean(self.hist[self.index_i - args.print_freq:self.index_i - 1, 0]) * args.print_freq,
                        np.mean(self.hist[self.index_i - args.print_freq:self.index_i - 1, 1]),
                        np.mean(self.hist[self.index_i - args.print_freq:self.index_i - 1, 2]),
                        np.mean(self.hist[self.index_i - args.print_freq:self.index_i - 1, 3]),
                        np.mean(self.hist[self.index_i - args.print_freq:self.index_i - 1, 4])
                    ), self.log
                )

    # One epoch's validation
    def validation(self, epoch):
        word_vocab = self.word_vocab
        self.decoder.eval()  # eval mode (no dropout or batchnorm)
        self.encoder_trans.eval()
        if self.encoder is not None:
            self.encoder.eval()

        val_start_time = time.time()
        references = list()  # references (true captions) for calculating BLEU-4 score
        hypotheses = list()  # hypotheses (predictions)

        self.evaluator.reset()
        with torch.no_grad():
            # Batches
            for ind, (imgA, imgB, seg_label, token_all, token_all_len, _, _, _) in enumerate(
                    tqdm(self.val_loader, desc='val_' + "EVALUATING AT BEAM SIZE " + str(1))):
                # Move to GPU, if available
                imgA = imgA.cuda()
                imgB = imgB.cuda()
                token_all = token_all.squeeze(0).cuda()
                # Forward prop.
                if self.encoder is not None:
                    feat1, feat2 = self.encoder(imgA, imgB)
                feat1, feat2 = self.Suppression_module(feat1, feat2)  # NEW
                feat1, feat2 = self.Complementary_module(feat1, feat2)  # NEW
                feat1, feat2, seg_pre, _ = self.encoder_trans(feat1, feat2)

                if self.args.train_goal != 0 or self.start_train_goal == 2:
                    seq = self.decoder.sample(feat1, feat2, k=1, seg_pre=seg_pre)

                # for segmentation
                if self.args.train_goal != 1  or self.start_train_goal == 2:
                    pred_seg = seg_pre.data.cpu().numpy()
                    seg_label = seg_label.cpu().numpy()
                    pred_seg = np.argmax(pred_seg, axis=1)
                    # Add batch sample into evaluator
                    self.evaluator.add_batch(seg_label, pred_seg)
                # for captioning
                if self.args.train_goal != 0 or self.start_train_goal == 2:
                    img_token = token_all.tolist()
                    img_tokens = list(map(lambda c: [w for w in c if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}],
                            img_token))  # remove <start> and pads
                    references.append(img_tokens)

                    pred_seq = [w for w in seq if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
                    hypotheses.append(pred_seq)
                    assert len(references) == len(hypotheses)

                    if ind % self.args.print_freq == 0:
                        pred_caption = ""
                        ref_caption = ""
                        for i in pred_seq:
                            pred_caption += (list(word_vocab.keys())[i]) + " "
                        ref_caption = ""
                        for i in img_tokens:
                            for j in i:
                                ref_caption += (list(word_vocab.keys())[j]) + " "
                            ref_caption += ".    "
            val_time = time.time() - val_start_time
            # Fast test during the training
            # for segmentation
            if self.args.train_goal != 1 or self.start_train_goal == 2:
                Acc_seg = self.evaluator.Pixel_Accuracy()
                Acc_class_seg = self.evaluator.Pixel_Accuracy_Class()
                mIoU_seg, IoU = self.evaluator.Mean_Intersection_over_Union()
                FWIoU_seg = self.evaluator.Frequency_Weighted_Intersection_over_Union()
                print_log(
                    '\nDetection_Validation:\n' 'Acc_seg: {0:.5f}\t' 'Acc_class_seg: {1:.5f}\t' 'mIoU_seg: {2:.5f}\t' 'FWIoU_seg: {3:.5f}\t '
                    .format(Acc_seg, Acc_class_seg, mIoU_seg, FWIoU_seg), self.log)
                print_log('Iou: {}'.format(IoU), self.log)

            # Calculate evaluation scores
            if self.args.train_goal != 0 or self.start_train_goal == 2:
                score_dict = get_eval_score(references, hypotheses)
                Bleu_1 = score_dict['Bleu_1']
                Bleu_2 = score_dict['Bleu_2']
                Bleu_3 = score_dict['Bleu_3']
                Bleu_4 = score_dict['Bleu_4']
                Meteor = score_dict['METEOR']
                Rouge = score_dict['ROUGE_L']
                Cider = score_dict['CIDEr']
                print_log('Captioning_Validation:\n' 'Time: {0:.3f}\t' 'BLEU-1: {1:.5f}\t' 'BLEU-2: {2:.5f}\t' 'BLEU-3: {3:.5f}\t' 
                    'BLEU-4: {4:.5f}\t' 'Meteor: {5:.5f}\t' 'Rouge: {6:.5f}\t' 'Cider: {7:.5f}\t'
                    .format(val_time, Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider), self.log)

        # Check if there was an improvement
        eps = 1e-6
        TH_MIOU = 0.83
        TH_BLEU4 = 0.65

        curr_bleu4 = locals().get('Bleu_4', None)
        curr_miou = locals().get('mIoU_seg', None)
        curr_sum = None if (curr_bleu4 is None or curr_miou is None) else (curr_bleu4 + curr_miou)

        # 1) Apply thresholds only to metrics computed in this validation pass.
        below_thresh = False
        th_reasons = []
        if curr_miou is not None and (curr_miou + eps) < TH_MIOU:
            below_thresh = True
            th_reasons.append(f'mIoU {curr_miou:.5f} < {TH_MIOU:.2f}')
        if curr_bleu4 is not None and (curr_bleu4 + eps) < TH_BLEU4:
            below_thresh = True
            th_reasons.append(f'BLEU-4 {curr_bleu4:.5f} < {TH_BLEU4:.2f}')

        if below_thresh:
            print_log(f"[CKPT] Skip: below thresholds ({' | '.join(th_reasons)})", self.log)
        else:
            # 2) Save when any metric, or the combined metric, improves over the best history.
            should_save = False
            reasons = []

            if curr_miou is not None and (curr_miou > self.MIou + eps):
                should_save = True
                reasons.append(f'mIoU {self.MIou:.5f} -> {curr_miou:.5f}')
            if curr_bleu4 is not None and (curr_bleu4 > self.best_bleu4 + eps):
                should_save = True
                reasons.append(f'BLEU-4 {self.best_bleu4:.5f} -> {curr_bleu4:.5f}')
            if curr_sum is not None and (curr_sum > self.Sum_Metric + eps):
                should_save = True
                reasons.append(f'Sum {self.Sum_Metric:.5f} -> {curr_sum:.5f}')

            if should_save:
                # Update best history only for metrics computed in this pass.
                if curr_bleu4 is not None:
                    self.best_bleu4 = max(self.best_bleu4, curr_bleu4)
                if curr_miou is not None:
                    self.MIou = max(self.MIou, curr_miou)
                if curr_sum is not None:
                    self.Sum_Metric = max(self.Sum_Metric, curr_sum)

                state = {
                    'encoder_dict': self.encoder.state_dict(),
                    'encoder_trans_dict': self.encoder_trans.state_dict(),
                    'decoder_dict': self.decoder.state_dict(),
                    'suppression_dict': self.Suppression_module.state_dict(),
                    'complementary_dict': self.Complementary_module.state_dict(),
                }
                if getattr(self, 'critic', None) is not None:
                    state['critic_dict'] = self.critic.state_dict()

                metric = f"Sum_{round(100000 * self.Sum_Metric)}_MIou_{round(100000 * self.MIou)}_Bleu4_{round(100000 * self.best_bleu4)}"
                model_name = f"{self.args.data_name}_bts_{self.args.train_batchsize}_{self.args.network}_epo_{epoch}_{metric}.pth"
                save_path = os.path.join(self.args.savepath, model_name)
                torch.save(state, save_path)

                self.best_epoch = epoch
                self.best_model_path = save_path

                print_log(f"[CKPT] Saved ({' | '.join(reasons)}) -> {model_name}", self.log)
            else:
                print_log("[CKPT] Skip: no improvement over best metrics after passing thresholds", self.log)


def set_seed(seed=42):
    random.seed(seed)                      # Python random module.
    np.random.seed(seed)                   # NumPy
    torch.manual_seed(seed)                # CPU
    torch.cuda.manual_seed(seed)           # Current GPU.
    torch.cuda.manual_seed_all(seed)       # All GPUs when using multi-GPU training.

    torch.backends.cudnn.deterministic = True   # Avoid nondeterministic algorithms.
    torch.backends.cudnn.benchmark = False      # Avoid run-to-run variation from autotuned kernels.



if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Interpretation')

    # Data parameters
    parser.add_argument('--sys', default='linux', help='system win or linux')
    parser.add_argument('--data_folder', default='./LEVIR-MCI-dataset/images', help='folder with data files')
    parser.add_argument('--list_path', default='./data/LEVIR_MCI/', help='path of the data lists')
    parser.add_argument('--token_folder', default='./data/LEVIR_MCI/tokens/', help='folder with token files')
    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--max_length', type=int, default=41, help='path of the data lists')
    parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    parser.add_argument('--data_name', default="LEVIR_MCI",help='base name shared by data files.')

    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint from stage s1, assert not None when train_stage=s2')
    parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches')
    # Training parameters
    parser.add_argument('--train_goal', type=int, default=2, help='0:det; 1:cap; 2:two tasks')
    parser.add_argument('--train_stage', default='s1', help='s1: pretrain backbone under two loss;'
                                                                         ' s2: train two branch respectively')
    parser.add_argument('--fine_tune_encoder', type=bool, default=True, help='whether fine-tune encoder or not')
    parser.add_argument('--train_batchsize', type=int, default=16, help='batch_size for training')
    parser.add_argument('--num_epochs', type=int, default=250, help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--workers', type=int, default=0, help='for data-loading')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for decoder.')
    parser.add_argument('--grad_clip', type=float, default=None, help='clip gradients at an absolute value of.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    # Validation
    parser.add_argument('--val_batchsize', type=int, default=1, help='batch_size for validation')
    parser.add_argument('--savepath', default="./models_ckpt/")
    # backbone parameters
    parser.add_argument('--network', default='segformer-mit_b1', help='define the backbone encoder to extract features')
    parser.add_argument('--encoder_dim', type=int, default=512,
                        help='the dimension of extracted features using backbone ')
    parser.add_argument('--feat_size', type=int, default=16,
                        help='define the output size of encoder to extract features')
    # Model parameters
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in AttentionEncoder.')
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--feature_dim', type=int, default=512, help='embedding dimension')
    parser.add_argument('--critic_in_ch', type=int, default=128)  # Channel count of cd_pred_features used by the Critic.
    parser.add_argument('--critic_lr', type=float, default=1e-4)
    parser.add_argument('--n_critic', type=int, default=1)
    parser.add_argument('--lambda_adv', type=float, default=0.3)
    parser.add_argument('--adv_warm_epochs', type=int, default=5)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--lambda_ent', type=float, default=1e-3)
    parser.add_argument('--lambda_tv', type=float, default=0.0)
    parser.add_argument('--lambda_align', type=float, default=0.1)
    parser.add_argument('--critic_grad_clip', type=float, default=5.0)
    args = parser.parse_args()

    trainer = Trainer(args)
    print('Starting Epoch:', trainer.start_epoch)
    print('Total Epoches:', trainer.args.num_epochs)

    if args.train_goal == 2:
        # First train both together, then train only change captioning, and finally train only change detection
        for goal in [2, 1, 0]:
            print_log(f'Current train_goal={goal}:\n', trainer.log)
            trainer.args.train_goal = goal
            if goal == 2:
                trainer.args.train_stage = 's1'
                trainer.args.checkpoint = None
                for epoch in range(trainer.start_epoch, trainer.args.num_epochs):
                    trainer.training(trainer.args, epoch)
                    trainer.validation(epoch)
                    if epoch - trainer.best_epoch > 50:
                        trainer.start_epoch = trainer.best_epoch + 1
                        break
                    elif epoch == trainer.args.num_epochs - 1:
                        trainer.start_epoch = trainer.best_epoch + 1
                        trainer.args.num_epochs = trainer.start_epoch + args.num_epochs
            else:
                trainer.args.train_stage = 's2'
                trainer.args.checkpoint = trainer.best_model_path
                trainer.build_model()
                for epoch in range(trainer.start_epoch, trainer.args.num_epochs):
                    trainer.training(trainer.args, epoch)
                    trainer.validation(epoch)
                    if trainer.args.train_goal == 1 and epoch - trainer.best_epoch > 50:
                        trainer.start_epoch = trainer.best_epoch + 1
                        trainer.args.num_epochs = trainer.start_epoch + trainer.args.num_epochs
                        break
                # trainer.args.num_epochs = trainer.start_epoch + trainer.args.num_epochs
    else:
        for epoch in range(trainer.start_epoch, trainer.args.num_epochs):
            trainer.training(trainer.args, epoch)
            # if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)
