import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM, BidirectionalLSTMv2
from modules.prediction import Attention
from modules.SVTR import SVTRNet
from modules.VIPTRv1 import VIPTRv1, VIPTRv1L
from modules.VIPTRv2 import VIPTRv2, VIPTRv2B
from modules.tps_spatial_transformer import TPSSpatialTransformer
from modules.stn_head import STNHead
from functools import partial

import argparse
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS17':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)

        elif opt.Transformation == 'TPS19':
            self.tps = TPSSpatialTransformer(output_image_size=[opt.imgH, opt.imgW],
                                             num_control_points=opt.num_fiducial,
                                             margins=[0.05, 0.05])
            self.stn_head = STNHead(in_planes=3, num_ctrlpoints=opt.num_fiducial, activation=None)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'VIPTRv1L':
            self.FeatureExtraction = VIPTRv1L(opt)
        elif opt.FeatureExtraction == 'VIPTRv1T':
            self.FeatureExtraction = VIPTRv1(opt)
        elif opt.FeatureExtraction == 'VIPTRv2T':
            self.FeatureExtraction = VIPTRv2(opt)
        elif opt.FeatureExtraction == 'VIPTRv2B':
            self.FeatureExtraction = VIPTRv2B(opt)
        elif opt.FeatureExtraction == 'SVTR':
            self.FeatureExtraction = SVTRNet(img_size=[32, opt.imgW], # 100
            in_channels=3,
            embed_dim=[64, 128, 256],
            depth=[3, 6, 3],
            num_heads=[2, 4, 8],
            mixer=['Local'] * 6 + ['Global'] * 6,  # Local atten, Global atten, Conv
            local_mixer=[[7, 11], [7, 11], [7, 11]],
            patch_merging='Conv',  # Conv, Pool, None
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            last_drop=0.1,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer='nn.LayerNorm',
            sub_norm='nn.LayerNorm',
            epsilon=1e-6,
            out_channels=opt.output_channel,
            out_char_num=opt.batch_max_length,  # 25
            block_unit='Block',
            act='nn.GELU',
            last_stage=True,
            sub_num=2,
            prenorm=False,
            use_lenhead=False,
            local_rank=device)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text=None, is_train=True):
        """ Transformation stage """
        if self.stages['Trans'] == "TPS17":
            stn_x = self.Transformation(input)
        elif self.stages['Trans'] == "TPS19":
            stn_input = F.interpolate(input, [32, 64], mode='bilinear', align_corners=True)
            _, ctrl_points = self.stn_head(stn_input)
            stn_x, _ = self.tps(input, ctrl_points)
        else:
            stn_x = input
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(stn_x)
        # visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        # visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='textMixer',
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='None', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='CTC', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=3,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=192,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()
    opt.num_class = 5961

    import time
    model = Model(opt).eval().cuda()
    print("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))
    x = torch.randn(2, 3, 32, 100).cuda()
    y = model(x)
    print(y.shape)

    # x = torch.randn(2, 3, 32, 320).cuda()
    # y = model(x)
    # print(y.shape)
    # start = time.time()
    # for i in range(100):
    #     # x = torch.randn(1, 3, 32, 1500).cuda()
    #     model(x)
    # print('GPU:', (time.time() - start) / 2)
    # x = torch.randn(1, 3, 32, 1500).cpu()
    # model.cpu()
    # model(x)
    # start = time.time()
    # for i in range(100):
    #     # x = torch.randn(1, 3, 32, 1500).cpu()
    #     model(x)
    # print('CPU:', (time.time() - start) / 2)
