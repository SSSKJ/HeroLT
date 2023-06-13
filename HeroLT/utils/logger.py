import os
import logging
from logging import handlers


def get_logger(LOG_ROOT, log_filename, level=logging.INFO, when='D', back_count=0):

    logger = logging.getLogger(log_filename)
    logger.setLevel(level)
    log_path = os.path.join(LOG_ROOT, "logs")
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_file_path = os.path.join(log_path, log_filename)
    formatter = logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(level)
    fh = handlers.TimedRotatingFileHandler(
        filename=log_file_path,
        when=when,
        backupCount=back_count,
        encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import yaml
import csv
import h5py


class Logger(object):
    def __init__(self, base_dir, model, dataset):
        self.logdir = f'{base_dir}/logs/{model}/dataset/'
        if not os.path.isdir(self.logdir):
            os.makedirs(self.logdir)
        self.cfg_file = os.path.join(self.logdir, 'cfg.yaml')
        self.acc_file = os.path.join(self.logdir, 'acc.csv')
        self.loss_file = os.path.join(self.logdir, 'loss.csv')
        self.ws_file = os.path.join(self.logdir, 'ws.h5')
        self.acc_keys = None
        self.loss_keys = None
        self.logging_ws = False
        self.log = get_logger(base_dir, f'{model}_{dataset}.log')

    def log_cfg(self, cfg):
        print('===> Saving cfg parameters to: ', self.cfg_file)
        with open(self.cfg_file, 'w') as f:
            yaml.dump(cfg, f)

    def log_acc(self, accs):
        if self.acc_keys is None:
            self.acc_keys = [k for k in accs.keys()]
            with open(self.acc_file, 'w') as f:
                writer = csv.DictWriter(f, fieldnames=self.acc_keys)
                writer.writeheader()
                writer.writerow(accs)
        else:
            with open(self.acc_file, 'a') as f:
                writer = csv.DictWriter(f, fieldnames=self.acc_keys)
                writer.writerow(accs)

    def log_loss(self, losses):
        # valid_losses = {k: v for k, v in losses.items() if v is not None}
        valid_losses = losses
        if self.loss_keys is None:
            self.loss_keys = [k for k in valid_losses.keys()]
            with open(self.loss_file, 'w') as f:
                writer = csv.DictWriter(f, fieldnames=self.loss_keys)
                writer.writeheader()
                writer.writerow(valid_losses)
        else:
            with open(self.loss_file, 'a') as f:
                writer = csv.DictWriter(f, fieldnames=self.loss_keys)
                writer.writerow(valid_losses)
    
    def log_ws(self, e, ws):
        mode = 'a' if self.logging_ws else 'w'
        self.logging_ws = True
        
        key = 'Epoch{:02d}'.format(e)
        with h5py.File(self.ws_file, mode) as f:
            g = f.create_group(key)
            for k, v in ws.items():
                g.create_dataset(k, data=v)
        