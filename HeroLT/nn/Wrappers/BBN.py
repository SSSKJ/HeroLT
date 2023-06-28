from . import BaseModel
from ...configs.BBN import _C as config
from ...configs.BBN import update_config
from ..Schedulers.WarmupMultiStepLR import WarmupMultiStepLR
from ..Models.BBN_Combiner import Combiner
from ..Models.BBN_Network import Network
from ...utils import accuracy, AverageMeter, FusionMatrix
from ...utils.logger import get_logger
from ..Datasets.BBNDataset import IMBALANCECIFAR10, IMBALANCECIFAR100, iNaturalist
from ..Loss.CSCE import CSCE
from ..Loss.CrossEntropy import CrossEntropy
from ..Loss.LDAMLoss import LDAMLoss


import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import numpy as np

import os
import time
from tqdm import tqdm

class BBN(BaseModel):


    def __init__(
            self,
            dataset: str,
            base_dir: str = '../../',
            ) -> None:
        
        super().__init__(
            model_name = 'BBN',
            dataset_name = dataset,
            base_dir = base_dir)
        
        self.__load_config()

        self.model_dir = os.path.join(self.output_path, self.config.NAME, "models") ## self.config.NAME??
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
                        
        self.logger = get_logger(self.base_dir, f'{self.model_name}_{self.dataset_name}.log')
        self.device = torch.device("cpu" if self.config.CPU_MODE else "cuda")
        self.train_set = None
        self.trainLoader = None
        self.validLoader = None
        self.valid_set = None
        self.testLoader = None
        self.test_set = None
        self.model = None

    def __load_config(self):

        config_path = f'{self.base_dir}/{self.model_name}/{self.dataset_name}/config.yaml'
        self.config = update_config(config, config_path)
        self.config.OUTPUT_DIR = self.output_path
        self.config.DATASET.ROOT = f'{self.base_dir}/data/CVData/'
    
    def load_data(self, phase = 'train'):

        if phase == 'train' and (self.trainLoader is None or self.validLoader is None):
                
            self.train_set = eval(self.config.DATASET.DATASET)("train", self.config)
            self.trainLoader = DataLoader(
                self.train_set,
                batch_size=self.config.TRAIN.BATCH_SIZE,
                shuffle=self.config.TRAIN.SHUFFLE,
                num_workers=self.config.TRAIN.NUM_WORKERS,
                pin_memory=self.config.PIN_MEMORY,
                drop_last=True
            )

            self.valid_set = eval(self.config.DATASET.DATASET)("valid", self.config)
            self.validLoader = DataLoader(
                self.valid_set,
                batch_size=self.config.TEST.BATCH_SIZE,
                shuffle=False,
                num_workers=self.config.TEST.NUM_WORKERS,
                pin_memory=self.config.PIN_MEMORY,
            )

        elif phase == 'eval' and (self.testLoader is None):

            self.test_set = eval(self.config.DATASET.DATASET)("valid", self.config)
            self.testLoader = DataLoader(
                self.test_set,
                batch_size=self.config.TEST.BATCH_SIZE,
                shuffle=False,
                num_workers=self.config.TEST.NUM_WORKERS,
                pin_memory=self.config.PIN_MEMORY,
            )

        else:
            raise RuntimeError(f'Unkown phase {phase}')

    def __init_model(self):
        
        self.model = Network(self.config, mode = "train", num_classes = self.num_classes)

        if self.config.BACKBONE.FREEZE == True:
            self.model.freeze_backbone()
            self.logger.info("Backbone has been freezed")

        if self.config.CPU_MODE:
            self.model = self.model.to(self.device)
        else:
            self.model = torch.nn.DataParallel(self.model).cuda()

        self.combiner = Combiner(self.config, self.device)

    def __init_optimizer_and_scheduler(self):

        base_lr = self.config.TRAIN.OPTIMIZER.BASE_LR
        params = []

        ## generate optimizer
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                params.append({"params": p})

        if self.config.TRAIN.OPTIMIZER.TYPE == "SGD":
            self.optimizer = torch.optim.SGD(
                params,
                lr=base_lr,
                momentum=self.config.TRAIN.OPTIMIZER.MOMENTUM,
                weight_decay=self.config.TRAIN.OPTIMIZER.WEIGHT_DECAY,
                nesterov=True,
            )
        elif self.config.TRAIN.OPTIMIZER.TYPE == "ADAM":
            self.optimizer = torch.optim.Adam(
                params,
                lr=base_lr,
                betas=(0.9, 0.999),
                weight_decay=self.config.TRAIN.OPTIMIZER.WEIGHT_DECAY,
            )
        
        ## generate scheduler
        if self.config.TRAIN.LR_SCHEDULER.TYPE == "multistep":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                self.config.TRAIN.LR_SCHEDULER.LR_STEP,
                gamma=self.config.TRAIN.LR_SCHEDULER.LR_FACTOR,
            )
        elif self.config.TRAIN.LR_SCHEDULER.TYPE == "cosine":
            if self.config.TRAIN.LR_SCHEDULER.COSINE_DECAY_END > 0:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.config.TRAIN.LR_SCHEDULER.COSINE_DECAY_END, eta_min=1e-4
                )
            else:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.config.TRAIN.MAX_EPOCH, eta_min=1e-4
                )
        elif self.config.TRAIN.LR_SCHEDULER.TYPE == "warmup":
            self.scheduler = WarmupMultiStepLR(
                self.optimizer,
                self.config.TRAIN.LR_SCHEDULER.LR_STEP,
                gamma=self.config.TRAIN.LR_SCHEDULER.LR_FACTOR,
                warmup_epochs=self.config.TRAIN.LR_SCHEDULER.WARM_EPOCH,
            )
        else:
            raise NotImplementedError("Unsupported LR Scheduler: {}".format(self.config.TRAIN.LR_SCHEDULER.TYPE))

    def train(self):

        cudnn.benchmark = True
        auto_resume = True

        self.annotations = self.train_set.get_annotations()
        self.num_classes = self.train_set.get_num_classes()

        num_class_list, cat_list = self.__get_category_list(self.annotations, self.num_classes)

        para_dict = {
            "self.num_classes": self.num_classes,
            "num_class_list": num_class_list,
            "self.config": self.config,
            "device": self.device,
        }

        criterion = eval(self.config.LOSS.LOSS_TYPE)(para_dict=para_dict)
        epoch_number = self.config.TRAIN.MAX_EPOCH

        # ----- BEGIN MODEL BUILDER -----
        self.__init_model()
        self.__init_optimizer_and_scheduler()
        # ----- END MODEL BUILDER -----

        self.load_data()

        best_result, best_epoch, start_epoch = self.load_pretrained_model()

        self.logger.info(
            "-------------------Train start :{}  {}  {}-------------------".format(
                self.config.BACKBONE.TYPE, self.config.MODULE.TYPE, self.config.TRAIN.COMBINER.TYPE
            )
        )

        for epoch in range(start_epoch, epoch_number + 1):
            self.scheduler.step()

            if self.config.EVAL_MODE:
                self.model.eval()
            else:
                self.model.train()

            self.combiner.reset_epoch(epoch)

            if self.config.LOSS.LOSS_TYPE in ['LDAMLoss', 'CSCE']:
                criterion.reset_epoch(epoch)

            start_time = time.time()
            number_batch = len(self.trainLoader)

            all_loss = AverageMeter()
            acc = AverageMeter()
            for i, (image, label, meta) in enumerate(self.trainLoader):
                cnt = label.shape[0]
                loss, now_acc = self.combiner.forward(self.model, criterion, image, label, meta)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                all_loss.update(loss.data.item(), cnt)
                acc.update(now_acc, cnt)

                if i % self.config.SHOW_STEP == 0:
                    pbar_str = "Epoch:{:>3d}  Batch:{:>3d}/{}  Batch_Loss:{:>5.3f}  Batch_Accuracy:{:>5.2f}%     ".format(
                        epoch, i, number_batch, all_loss.val, acc.val * 100
                    )
                    self.logger.info(pbar_str)
            end_time = time.time()
            pbar_str = "---Epoch:{:>3d}/{}   Avg_Loss:{:>5.3f}   Epoch_Accuracy:{:>5.2f}%   Epoch_Time:{:>5.2f}min---".format(
                epoch, epoch_number, all_loss.avg, acc.avg * 100, (end_time - start_time) / 60
            )
            self.logger.info(pbar_str)
            train_acc = acc.avg
            train_loss = all_loss.avg

            model_save_path = os.path.join(
                self.model_dir,
                "epoch_{}.pth".format(epoch),
            )
            if epoch % self.config.SAVE_STEP == 0:
                torch.save({
                    'state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'best_result': best_result,
                    'best_epoch': best_epoch,
                    'scheduler': self.scheduler.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }, model_save_path)

            loss_dict, acc_dict = {"train_loss": train_loss}, {"train_acc": train_acc}
            if self.config.VALID_STEP != -1 and epoch % self.config.VALID_STEP == 0:
                valid_acc, valid_loss = self.__valid_model(self.validLoader, epoch, self.model, self.config, criterion)
                loss_dict["valid_loss"], acc_dict["valid_acc"] = valid_loss, valid_acc
                if valid_acc > best_result:
                    best_result, best_epoch = valid_acc, epoch
                    torch.save({
                            'state_dict': self.model.state_dict(),
                            'epoch': epoch,
                            'best_result': best_result,
                            'best_epoch': best_epoch,
                            'scheduler': self.scheduler.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                    }, os.path.join(self.model_dir, "best_model.pth")
                    )
                self.logger.info(
                    "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                        best_epoch, best_result * 100
                    )
                )
        self.logger.info(
            "-------------------Train Finished :{}-------------------".format(self.config.NAME)
        )

    def __get_category_list(self):
        num_list = [0] * self.num_classes
        cat_list = []
        self.logger.info("Weight List has been produced")
        for anno in self.annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
            cat_list.append(category_id)
        return num_list, cat_list

    def __valid_model(self, dataLoader, epoch_number, criterion):

        self.model.eval()
        num_classes = dataLoader.dataset.get_num_classes()
        fusion_matrix = FusionMatrix(num_classes)


        with torch.no_grad():
            all_loss = AverageMeter()
            acc = AverageMeter()
            func = torch.nn.Softmax(dim=1)
            for i, (image, label, meta) in enumerate(dataLoader):
                image, label = image.to(self.device), label.to(self.device)

                feature = self.model(image, feature_flag=True)

                output = self.model(feature, classifier_flag=True)
                loss = criterion(output, label)
                score_result = func(output)

                now_result = torch.argmax(score_result, 1)
                all_loss.update(loss.data.item(), label.shape[0])
                fusion_matrix.update(now_result.cpu().numpy(), label.cpu().numpy())
                now_acc, cnt = accuracy(now_result.cpu().numpy(), label.cpu().numpy())
                acc.update(now_acc, cnt)

            pbar_str = "------- Valid: Epoch:{:>3d}  Valid_Loss:{:>5.3f}   Valid_Acc:{:>5.2f}%-------".format(
                epoch_number, all_loss.avg, acc.avg * 100
            )
            self.logger.info(pbar_str)
        return acc.avg, all_loss.avg
    
    def load_pretrained_model(self):
        # ----- BEGIN RESUME ---------
        all_models = os.listdir(self.model_dir)
        if len(all_models) <= 1 or auto_resume == False:
            auto_resume = False
        else:
            all_models.remove("best_model.pth")
            resume_epoch = max([int(name.split(".")[0].split("_")[-1]) for name in all_models])
            resume_model_path = os.path.join(self.model_dir, "epoch_{}.pth".format(resume_epoch))

        if self.config.RESUME_MODEL != "" or auto_resume:
            if self.config.RESUME_MODEL == "":
                resume_model = resume_model_path
            else:
                resume_model = self.config.RESUME_MODEL if '/' in self.config.RESUME_MODEL else os.path.join(model_dir, self.config.RESUME_MODEL)
            self.logger.info("Loading checkpoint from {}...".format(resume_model))
            checkpoint = torch.load(
                resume_model, map_location="cpu" if self.config.CPU_MODE else "cuda"
            )
            if self.config.CPU_MODE:
                self.model.load_model(resume_model)
            else:
                self.model.module.load_model(resume_model)
            if self.config.RESUME_MODE != "state_dict":
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                return checkpoint['epoch'] + 1, checkpoint['best_result'], checkpoint['best_epoch']

        return 0, 0, 1
        # ----- END RESUME ---------

    def eval(self):

        self.load_data(phase='eval')

        num_classes = self.test_set.get_num_classes()
        self.model = Network(self.config, mode="test", num_classes=num_classes)

        model_dir = os.path.join(self.config.OUTPUT_DIR, self.config.NAME, "models")
        model_file = self.config.TEST.MODEL_FILE
        if "/" in model_file:
            model_path = model_file
        else:
            model_path = os.path.join(model_dir, model_file)
        model.load_model(model_path)

        if self.config.CPU_MODE:
            model = model.to(self.device)
        else:
            model = torch.nn.DataParallel(model).cuda()

        result_list = []
        pbar = tqdm(total=len(self.testLoader))
        model.eval()
        top1_count, top2_count, top3_count, index, fusion_matrix = (
            [],
            [],
            [],
            0,
            FusionMatrix(num_classes),
        )

        func = torch.nn.Softmax(dim=1)

        with torch.no_grad():
            for i, (image, image_labels, meta) in enumerate(self.testLoader):
                image = image.to(self.device)
                output = model(image)
                result = func(output)
                _, top_k = result.topk(5, 1, True, True)
                score_result = result.cpu().numpy()
                fusion_matrix.update(score_result.argmax(axis=1), image_labels.numpy())
                topk_result = top_k.cpu().tolist()
                if not "image_id" in meta:
                    meta["image_id"] = [0] * image.shape[0]
                image_ids = meta["image_id"]
                for i, image_id in enumerate(image_ids):
                    result_list.append(
                        {
                            "image_id": image_id,
                            "image_label": int(image_labels[i]),
                            "top_3": topk_result[i],
                        }
                    )
                    top1_count += [topk_result[i][0] == image_labels[i]]
                    top2_count += [image_labels[i] in topk_result[i][0:2]]
                    top3_count += [image_labels[i] in topk_result[i][0:3]]
                    index += 1
                now_acc = np.sum(top1_count) / index
                pbar.set_description("Now Top1:{:>5.2f}%".format(now_acc * 100))
                pbar.update(1)
        top1_acc = float(np.sum(top1_count) / len(top1_count))
        top2_acc = float(np.sum(top2_count) / len(top1_count))
        top3_acc = float(np.sum(top3_count) / len(top1_count))
        self.logger.info(
            "Top1:{:>5.2f}%  Top2:{:>5.2f}%  Top3:{:>5.2f}%".format(
                top1_acc * 100, top2_acc * 100, top3_acc * 100
            )
        )
        pbar.close()
