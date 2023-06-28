from .BaseModel import BaseModel
from ..pecos.xmc.xtransformer.model import XTransformer
from ..pecos.xmc.xtransformer.module import MLProblemWithText
from ..pecos.utils import smat_util, torch_util
from ..pecos.utils.featurization.text.preprocess import Preprocessor
from ..pecos.xmc import Indexer, LabelEmbeddingFactory
from ..pecos.utils.smat_util import sorted_csr, CsrEnsembler, load_matrix
from ...utils.logger import get_logger

import numpy as np

import os
import json
import time
import gc

class XRTransformer(BaseModel):


    def __init__(
            self,
            dataset: str,
            base_dir: str = '../../',
            ) -> None:
        
        super().__init__(
            model_name = 'XRTransformer',
            dataset_name = dataset,
            base_dir = base_dir)
        
        super().load_config()

        self.model_dir = None
        self.param_path = None

        self.X_trn = None
        self.Y_trn = None
        self.trn_corpus = None

        self.X_tst = None
        self.tst_corpus = None

        self.xtf = None

        self.data_dir=f'{self.base_dir}/data/NLPData/xmc-base/{self.dataset_name}'

        self.logger = get_logger(self.base_dir, f'{self.model_name}_{self.dataset_name}.log')
        

    def train(self):

        self.data_dir=f'{self.base_dir}/data/xmc-base/{self.dataset_name}'

        if self.dataset_name == "eurlex-4k":
            self.models = ['bert', 'roberta', 'xlnet']
            self.ens_method = 'softmax_average'
        elif self.dataset_name == "wiki10-31k":
            models = ['bert']
            ens_method = 'rank_average'
        elif self.dataset_name == "amazoncat-13k":
            models = ['bert', 'roberta', 'xlnet']
            ens_method = 'softmax_average'
        elif self.dataset_name == "wiki-500k":
            models = ['bert1' ,'bert2', 'bert3']
            ens_method = 'sigmoid_average'
        elif self.dataset_name == "amazon-670k":
            models = ['bert1' ,'bert2', 'bert3']
            ens_method = 'softmax_average'
        elif self.dataset_name == "amazon-3m":
            models = ['bert1' ,'bert2', 'bert3']
            ens_method = 'rank_average'
        else:
            raise RuntimeError(f'Unkonwn Dataset {self.dataset_name}')

        for model in models:

            self.model_dir = f'{self.output_path}/{model}'
            os.makedirs(self.model_dir, exist_ok = True)
            self.param_path = f'{self.base_dir}/configs/{self.model_name}/params/{self.dataset_name}/{model}/params.json'

            self.__train_model()
            self.__predict()
            self.__evaluate()

        self.__ensemble_evaluate()
    
    def load_data(self, phase):
        
        if phase == 'train':
            
            if (self.X_trn is None or self.Y_trn is None or self.trn_corpus is None):
                self.logger.info("| loading data begin...")
                start_time = time.time()
                # Load training feature
                if os.path.exists(f'{self.data_dir}/tfidf-attnxml/X.trn.npz'):
                    self.X_trn = smat_util.load_matrix(f'{self.data_dir}/tfidf-attnxml/X.trn.npz', dtype=np.float32)
                    self.logger.info("Loaded training feature matrix with shape={}".format(self.X_trn.shape))
                else:
                    raise RuntimeError(f'Can\'t find any Training Data, please check if {self.data_dir}/tfidf-attnxml/X.trn.npz exists')

                # Load training labels
                if os.path.exists(f'{self.data_dir}/Y.trn.npz'):
                    self.Y_trn = smat_util.load_matrix(f'{self.data_dir}/Y.trn.npz', dtype=np.float32)
                    self.logger.info("Loaded training label matrix with shape={}".format(self.Y_trn.shape))
                else:
                    raise RuntimeError(f'Can\'t find any Training Data, please check if {self.data_dir}/Y.trn.npz exists')
                
                if os.path.exists(f'{self.data_dir}/X.trn.txt'):
                    # Load training texts
                    self.trn_corpus = Preprocessor.load_data_from_file(
                        f'{self.data_dir}/X.trn.txt',
                        label_text_path=None,
                        text_pos=0,
                    )["corpus"]
                    self.logger.info("Loaded {} training sequences".format(len(self.trn_corpus)))
                else:
                    raise RuntimeError(f'Can\'t find any Training Data, please check if {self.data_dir}/X.trn.txt exists')
                
                run_time_io = time.time() - start_time
                self.logger.info("| loading data finsihed | time(s) {:9.4f}".format(run_time_io))
                self.logger.info('Finish Loading Data for training')
            
            else:

                self.logger.info('Data for training was loaded')

        elif phase == 'predict':
            
            if (self.X_tst is None or self.tst_corpus is None):

                # Load data
                self.logger.info("| loading data begin...")
                start_time = time.time()

                # load instance feature and text
                if os.path.exists(f'{self.data_dir}/tfidf-attnxml/X.tst.npz'):
                    self.X_tst = smat_util.load_matrix(f'{self.data_dir}/tfidf-attnxml/X.tst.npz')
                    self.logger.info("Loaded testing feature matrix with shape={}".format(self.Y_trn.shape))
                else:
                    raise RuntimeError(f'Can\'t find any testing Data, please check if {self.data_dir}/tfidf-attnxml/X.tst.npz exists')
                
                if os.path.exists(f'{self.data_dir}/X.tst.txt'):
                    self.tst_corpus = Preprocessor.load_data_from_file(f'{self.data_dir}/X.tst.txt', label_text_path=None, text_pos=0)["corpus"]
                else:
                    raise RuntimeError(f'Can\'t find any testing Data, please check if {self.data_dir}/X.tst.txt exists')
                
                run_time_data = time.time() - start_time
                self.logger.info("| loading data finsihed | time(s) {:9.4f}".format(run_time_data))
                self.logger.info('Finish Loading Data for prediction')
            
            else:

                self.logger.info('Data for prediction was loaded')
        else:
            raise RuntimeError(f'Unkown phase {phase}')



    def __train_model(self):
        """Train and save XR-Transformer model.

        Args:
            args (argparse.Namespace): Command line arguments parsed by `parser.parse_args()`
        """

        config = self.config['train']
        params = dict()

        if os.path.exists(self.param_path):
            with open(self.param_path, "r") as file:
                params = json.load(file)

        train_params = params.get("train_params", None)
        pred_params = params.get("pred_params", None)

        if train_params is not None:
            train_params = XTransformer.TrainParams.from_dict(train_params)
        else:
            train_params = XTransformer.TrainParams.from_dict(
                {k: v for k, v in config.items() if v is not None},
                recursive=True,
            )

        if pred_params is not None:
            pred_params = XTransformer.PredParams.from_dict(pred_params)
        else:
            pred_params = XTransformer.PredParams.from_dict(
                {k: v for k, v in config.items() if v is not None},
                recursive=True,
            )

        torch_util.set_seed(config['seed'])
        self.logger.info("Setting random seed {}".format(config['seed']))

        self.load_data('train')

        # load cluster chain
        label_feat = LabelEmbeddingFactory.pifa(self.Y_trn, self.X_trn)

        cluster_chain = Indexer.gen(
            label_feat,
            logger=self.logger,
            train_params=train_params.preliminary_indexer_params,
        )
        del label_feat
        gc.collect()

        trn_prob = MLProblemWithText(self.trn_corpus, self.Y_trn, X_feat = self.X_trn)

        self.xtf = XTransformer.train(
            trn_prob,
            clustering=cluster_chain,
            val_prob=None,
            train_params=train_params,
            pred_params=pred_params,
            beam_size=None,
            logger=self.logger,
        )

        self.xtf.save(self.model_dir, logger = self.logger)
    
    def load_pretrained_model(self):

        self.xtf = XTransformer.load(self.model_dir, logger = self.logger)

    def __predict(self):

        config = self.config['predict']
        """Predict with XTransformer and save the result.

        Args:
            args (argparse.Namespace): Command line arguments parsed by `parser.parse_args()`
        """

        torch_util.set_seed(config['seed'])
        self.load_data('predict')

        if self.xtf is None:

            self.load_pretrained_model()

        P_matrix = self.xtf.predict(
            self.tst_corpus,
            X_feat=self.X_tst,
            batch_size=config['batch_size'],
            batch_gen_workers=config['batch_gen_workers'],
            use_gpu=config['use_gpu'],
            max_pred_chunk=config['max_pred_chunk'],
            threads=config['threads'],
            logger = self.logger,
        )

        smat_util.save_matrix(os.path.join(self.model_dir, "Pt.npz"), P_matrix)

    def __evaluate(self):

        config = self.config['eval']

        Y_true = smat_util.load_matrix(f'{self.data_dir}/Y.tst.npz').tocsr()
        Y_pred = smat_util.load_matrix(f'{self.model_dir}/Pt.npz').tocsr()
        metric = smat_util.Metrics.generate(Y_true, Y_pred, topk=config['topk'])
        self.logger.log("==== evaluation results ====")
        self.logger.log(metric)

    def __ensemble_evaluate(self, ens_method):

        Y_true = sorted_csr(load_matrix(f'{self.data_dir}/Y.tst.npz').tocsr())
        Y_pred = [sorted_csr(load_matrix(pp).tocsr()) for pp in [f'{self.output_path}/{model}/Pt.npz' for model in self.models]]
        self.logger.log("==== evaluation results ====")
        CsrEnsembler.print_ens(Y_true, Y_pred, self.models, logger = self.logger, ens_method=self.ens_method)