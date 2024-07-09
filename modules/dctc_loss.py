import math
from typing import List
from torch import Tensor, nn
from torch.nn import functional as F
import torch

class DCTC(nn.Module):
    def __init__(self,
                 flatten: bool = True,
                 blank: int = 0,
                 reduction: str = 'mean',
                 s: float = 1,
                 m: float = 0,
                 alpha: float = 0.1,
                 beta: float = 1.0,
                 eps: float = 1e-8,
                 use_il: bool = True,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.flatten = flatten
        self.reduction = reduction
        self.s = s
        self.m = m
        self.scaled_margin = s * m
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.black = blank

        self.use_il = use_il

        self.ctc_loss_func = nn.CTCLoss(blank=blank, reduction='none', zero_infinity=True)
        self.ctc_loss_func_dummy = nn.CTCLoss(blank=blank, reduction='none', zero_infinity=True)

    def forward(self,
                logits: Tensor,
                targets_dict: dict,
                valid_ratios: List[float] = None
                ):
        alpha = self.alpha
        beta = self.beta
        seq_len, bs, v = logits.size()
        scaled_margin = self.scaled_margin

        if self.flatten:
            targets = targets_dict['targets']
        else:
            targets = torch.full(size=(bs, seq_len), fill_value=self.blank, dtype=torch.long)
            for idx, tensor in enumerate(targets_dict['targets']):
                valid_len = min(tensor.size(0), seq_len)
                targets[idx, :valid_len] = tensor[:valid_len]

        logits = self.s * logits

        target_lengths = targets_dict['target_lengths']

        if not self.use_il:
            valid_ratios = [1.0] * bs
        else:
            if valid_ratios is None:
                raise ValueError('Valid ratios should not be none, if use_il is True.')

        input_lengths = [int(math.ceil(seq_len * r)) for r in valid_ratios]
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)

        log_probs = torch.log_softmax(logits, 2)
        ctc_loss_1 = self.ctc_loss_func(log_probs, targets, input_lengths, target_lengths)

        if alpha > 0:
            with torch.enable_grad():
                log_probs_dummy = log_probs.detach().clone()
                log_probs_dummy.requires_grad = True
                ctc_loss_2 = self.ctc_loss_func_dummy(log_probs_dummy, targets, input_lengths, target_lengths)

                ctc_loss_2.sum().backward()

            grad = log_probs_dummy.grad

            with torch.no_grad():
                classes = torch.argmin(
                    grad / torch.clip(torch.softmax(logits, dim=2), min=self.eps),
                    dim=2
                )

            one_hots = F.one_hot(classes, v).to(logits.device).float()
            neg_log_margin_probs = -torch.log_softmax(logits - scaled_margin * one_hots, 2)
            selected_neg_log_margin_probs = neg_log_margin_probs * one_hots

            if self.use_il:
                il_mask = torch.arange(seq_len, device=logits.device)[..., None]
                il_mask = torch.ge(il_mask, input_lengths[None, ...].to(logits.device))
                il_mask = il_mask[..., None]
                selected_neg_log_margin_probs = torch.where(
                    il_mask,
                    torch.zeros_like(selected_neg_log_margin_probs),
                    selected_neg_log_margin_probs
                )

            ce_loss = selected_neg_log_margin_probs.sum(2).sum(0)
        else:
            ce_loss = 0

        nll = alpha * ce_loss + beta * ctc_loss_1

        if self.reduction == 'mean':
            loss = nll.mean()
        elif self.reduction == 'sum':
            loss = nll.sum()
        else:
            loss = nll

        return loss