"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from .ResNetFeature import *
from ...utils import init_weights
from os import path
        
def create_model(logger, use_selfatt=False, use_fc=False, dropout=None, stage1_weights=False, dataset=None, log_dir=None, test=False, *args):
    
    logger.info('Loading Scratch ResNet 152 Feature Model.')
    resnet = ResNet(Bottleneck, [3, 8, 36, 3], use_modulatedatt=use_selfatt, use_fc=use_fc, dropout=None)

    if not test:
        if stage1_weights:
            assert(dataset)
            logger.info('Loading %s Stage 1 ResNet 152 Weights.' % dataset)
            if log_dir is not None:
                weight_dir = path.join('/'.join(log_dir.split('/')[:-1]), 'stage1')
            else:
                weight_dir = './logs/%s/stage1' % dataset
            logger.info('==> Loading weights from %s' % weight_dir)
            resnet = init_weights(model=resnet,
                                    weights_path=path.join(weight_dir, 'final_model_checkpoint.pth'))
        else:
            logger.info('No Pretrained Weights For Feature Model.')

    return resnet
