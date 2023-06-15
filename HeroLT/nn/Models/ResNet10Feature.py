"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""


from .ResNetFeature import *
from ...utils import init_weights
from os import path

def create_model(logger, use_selfatt=False, use_fc=False, dropout=None, stage1_weights=False, dataset=None, log_dir=None, test=False, *args):
    
    logger.info('Loading Scratch ResNet 10 Feature Model.')
    resnet10 = ResNet(BasicBlock, [1, 1, 1, 1], use_modulatedatt=use_selfatt, use_fc=use_fc, dropout=None)

    if not test:
        if stage1_weights:
            assert(dataset)
            logger.info('Loading %s Stage 1 ResNet 10 Weights.' % dataset)
            if log_dir is not None:
                weight_dir = path.join('/'.join(log_dir.split('/')[:-1]), 'stage1')
            else:
                weight_dir = './logs/%s/stage1' % dataset
            logger.info('==> Loading weights from %s' % weight_dir)
            resnet10 = init_weights(model=resnet10,
                                    weights_path=path.join(weight_dir, 'final_model_checkpoint.pth'))
        else:
            logger.info('No Pretrained Weights For Feature Model.')

    return resnet10
