import copy
import torch
import math
from functools import partial
from torch.optim import lr_scheduler
import importlib
from torch.autograd import Variable
import numpy as np

#
# __all__ = ['build_optimizer']
#
#
# def build_optimizer(optim_config, lr_scheduler_config, epochs, step_each_epoch, model):
#     from . import lr
#     config = copy.deepcopy(optim_config)
#     optim = getattr(torch.optim, config.pop('name'))(params=model.parameters(), **config)
#
#     lr_config = copy.deepcopy(lr_scheduler_config)
#     lr_config.update({'epochs': epochs, 'step_each_epoch': step_each_epoch})
#     lr_scheduler = getattr(lr, lr_config.pop('name'))(**lr_config)(optimizer=optim)
#     return optim, lr_scheduler

class StepLR(object):
    def __init__(self,
                 step_each_epoch,
                 step_size,
                 warmup_epoch=0,
                 gamma=0.1,
                 last_epoch=-1,
                 **kwargs):
        super(StepLR, self).__init__()
        self.step_size = step_each_epoch * step_size
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch

    def __call__(self, optimizer):
        return lr_scheduler.LambdaLR(optimizer, self.lambda_func, self.last_epoch)

    def lambda_func(self, current_step):
        if current_step < self.warmup_epoch:
            return float(current_step) / float(max(1, self.warmup_epoch))
        return self.gamma ** (current_step // self.step_size)


class MultiStepLR(object):
    def __init__(self,
                 step_each_epoch,
                 milestones,
                 warmup_epoch=0,
                 gamma=0.1,
                 last_epoch=-1,
                 **kwargs):
        super(MultiStepLR, self).__init__()
        self.milestones = [step_each_epoch * e for e in milestones]
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch

    def __call__(self, optimizer):
        return lr_scheduler.LambdaLR(optimizer, self.lambda_func, self.last_epoch)

    def lambda_func(self, current_step):
        if current_step < self.warmup_epoch:
            return float(current_step) / float(max(1, self.warmup_epoch))
        return self.gamma ** len([m for m in self.milestones if m <= current_step])

class ConstLR(object):
    def __init__(self,
                 step_each_epoch,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        super(ConstLR, self).__init__()
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch * step_each_epoch

    def __call__(self, optimizer):
        return lr_scheduler.LambdaLR(optimizer, self.lambda_func, self.last_epoch)

    def lambda_func(self, current_step):
        if current_step < self.warmup_epoch:
            return float(current_step) / float(max(1.0, self.warmup_epoch))
        return 1.0


class LinearLR(object):
    def __init__(self,
                 epochs,
                 step_each_epoch,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        super(LinearLR, self).__init__()
        self.epochs = epochs * step_each_epoch
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch * step_each_epoch

    def __call__(self, optimizer):
        return lr_scheduler.LambdaLR(optimizer, self.lambda_func, self.last_epoch)

    def lambda_func(self, current_step):
        if current_step < self.warmup_epoch:
            return float(current_step) / float(max(1, self.warmup_epoch))
        return max(0.0, float(self.epochs - current_step) / float(max(1, self.epochs - self.warmup_epoch)))


class CosineAnnealingLR(object):
    def __init__(self,
                 epochs,
                 step_each_epoch,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        super(CosineAnnealingLR, self).__init__()
        self.epochs = epochs * step_each_epoch
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch * step_each_epoch

    def __call__(self, optimizer):
        return lr_scheduler.LambdaLR(optimizer, self.lambda_func, self.last_epoch)

    def lambda_func(self, current_step, num_cycles=0.5):
        if current_step < self.warmup_epoch:
            return float(current_step) / float(max(1, self.warmup_epoch))
        progress = float(current_step - self.warmup_epoch) / float(max(1, self.epochs - self.warmup_epoch))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


class PolynomialLR(object):
    def __init__(self,
                 step_each_epoch,
                 epochs,
                 lr_end=1e-7,
                 power=1.0,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        super(PolynomialLR, self).__init__()
        self.lr_end = lr_end
        self.power = power
        self.epochs = epochs * step_each_epoch
        self.warmup_epoch = warmup_epoch * step_each_epoch
        self.last_epoch = last_epoch

    def __call__(self, optimizer):
        lr_lambda = partial(
            self.lambda_func,
            lr_init=optimizer.defaults["lr"],
        )
        return lr_scheduler.LambdaLR(optimizer, lr_lambda, self.last_epoch)

    def lambda_func(self, current_step, lr_init):
        if current_step < self.warmup_epoch:
            return float(current_step) / float(max(1, self.warmup_epoch))
        elif current_step > self.epochs:
            return self.lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - self.lr_end
            decay_steps = self.epochs - self.warmup_epoch
            pct_remaining = 1 - (current_step - self.warmup_epoch) / decay_steps
            decay = lr_range * pct_remaining ** self.power + self.lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init


def get_no_weight_decay_param(model, config):
    param_names = config['optimizer']['no_weight_decay_param']['param_names']
    weight_decay = config['optimizer']['no_weight_decay_param']['weight_decay']
    is_on = config['optimizer']['no_weight_decay_param']['is_ON']
    if not is_on:
        return model.parameters()
    base_param = []
    no_weight_decay_param = []
    for (name, param) in model.named_parameters():
        is_no_weight = False
        for param_name in param_names:
            if param_name in name:
                is_no_weight = True
                break
        if is_no_weight:
            no_weight_decay_param.append(param)
        else:
            base_param.append(param)
    Outparam = [{'params': base_param}, {'params': no_weight_decay_param, 'weight_decay': weight_decay}]
    return Outparam

def fix_param(model, opt):
    param_names = ['pos_embed', 'norm']  # config['optimizer']['no_weight_decay_param']['param_names']
    weight_decay = 0.  # config['optimizer']['no_weight_decay_param']['weight_decay']
    is_on = True  # config['optimizer']['no_weight_decay_param']['is_ON']
    STN_ON = True  # config['model']['STN']['STN_ON']
    stn_lr = opt.base_lr  # config['model']['STN']['stn_lr']

    base_param = []
    stn_param = []
    no_weight_decay_param = []
    for (name, param) in model.named_parameters():
        is_no_weight = False
        for param_name in param_names:
            if param_name in name:
                # print(param_name)
                is_no_weight = True
                break
        if is_no_weight:
            no_weight_decay_param.append(param)
        elif 'stn' in name:
            stn_param.append(param)
        else:
            base_param.append(param)
    Outparam = [{'params': base_param}, {'params': stn_param}, {'params': no_weight_decay_param}]

    if STN_ON:
        Outparam[1]['lr'] = stn_lr
    if is_on:
        Outparam[2]['weight_decay'] = weight_decay
    return Outparam

def lr_warm(base_lr, epoch, warm_epoch):
    return (base_lr/warm_epoch)*(epoch+1)

def adjust_learning_rate_warm(opt, optimizer, epoch):
    lr = lr_warm(opt.base_lr, epoch, 2)  # lr_warm(config['optimizer']['base_lr'], epoch,config['train']['warmepochs'])
    optimizer.param_groups[0]['lr'] = lr
    if 'TPS' in opt.Transformation:
        stn_lr = opt.base_lr  # config['model']['STN']['stn_lr']
        lr = lr_warm(stn_lr, epoch, 2)  # lr_warm(stn_lr, epoch,config['train']['warmepochs'])
        optimizer.param_groups[1]['lr'] = lr


def adjust_learning_rate_cos(opt, optimizer, epoch):
    initial_learning_rate, step, decay_steps, alpha = opt.base_lr, epoch - 2, opt.num_epochs - 2, 0
    step = min(step, decay_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    optimizer.param_groups[0]['lr'] = initial_learning_rate * decayed

    if 'TPS' in opt.Transformation:
        stn_lr = opt.base_lr * decayed  # config['model']['STN']['stn_lr'] * decayed
        optimizer.param_groups[1]['lr'] = stn_lr