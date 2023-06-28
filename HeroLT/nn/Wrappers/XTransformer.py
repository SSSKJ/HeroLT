from . import BaseModel
from ..Models.LinearModel import LinearModel, solver_dict
from ..xbert.rf_linear import Metrics, HierarchicalMLModel, PostProcessor, LabelEmbeddingFactory
from ..xbert.indexer import Indexer
from ..xbert import rf_linear
from ...utils.logger import get_logger

from sklearn.preprocessing import normalize as sk_normalize
import scipy.sparse as smat

import os

class XTransformer(BaseModel):


    def __init__(
            self,
            dataset: str,
            base_dir: str = '../../',
            ) -> None:
        
        super().__init__(
            model_name = 'XTransformer',
            dataset_name = dataset,
            base_dir = base_dir)
        
        super().load_config()

        self.output_dir = None
        self.matcher_dir = None
        self.ranker_dir = None

        self.X = None
        self.Y = None
        self.C = None
        self.Z_pred = None

        self.Xt = None
        self.Yt = None
        self.csr_codes = None

        self.logger = get_logger(self.base_dir, f'{self.model_name}_{self.dataset_name}.log')

        self.label_name_arr = ['pifa-tfidf-s0', 'pifa-neural-s0', 'text-emb-s0']
        self.model_name_arr = ['bert-large-cased-whole-word-masking', 'roberta-large', 'xlnet-large-cased']
        self.data_dir = f'{self.base_dir}/data/NLPData/{self.dataset_name}'
        

    def train(self):

        for label_name in self.label_name_arr:

            self.output_dir = f'{self.output_path}/{label_name}'
            
            for model_name in self.model_name_arr:

                self.matcher_dir = f'{self.output_dir}/matcher/{model_name}'
                self.ranker_dir = f'{self.output_dir}/ranker/{model_name}'

                os.makedirs(self.ranker_dir, exist_ok = True)
                
                self.logger.info(f'Training ranker for {model_name} with {label_name}')

                # train linear ranker
                self.__train_ranker()

                # predict final label ranking, using transformer's predicted cluster scores
                self.__predict()

        # final eval
        self.eval()


    def __train_ranker(self):

        config = self.config['train']

        self.load_data('train')

        model = LinearModel.train(
            self.X,
            self.Y,
            self.C,
            mode=config['mode'],
            solver_type=solver_dict[config['solver_type']],
            Cp=config['Cp'],
            Cn=config['Cn'],
            threshold=config['threshold'],
            threads=config['threads'],
            bias=config['bias'],
            Z_pred=self.Z_pred,
            negative_sampling_scheme=config['negative_sampling_scheme'],
        )

        model.save(self.ranker_dir)

    def load_data(self, phase):
            
        if phase == 'train':
            
            if (self.X is None or self.Y is None or self.C is None or self.Z_pred is None):

                if os.path.exists(f'{self.data_dir}/X.trn.npz'):
                    X1 = HierarchicalMLModel.load_feature_matrix(f'{self.data_dir}/X.trn.npz')
                else:
                    raise RuntimeError(f'Can\'t find any Training Data, please check if {self.data_dir}/X.trn.npz exists')
                
                if os.path.exists(f'{self.matcher_dir}/trn_embeddings.npy'):
                    X2 = HierarchicalMLModel.load_feature_matrix(f'{self.matcher_dir}/trn_embeddings.npy')
                else:
                    raise RuntimeError(f'Can\'t find any Training Data, please check if {self.matcher_dir}/trn_embeddings.npy exists')
                
                self.X = smat.hstack([sk_normalize(X1, axis=1), sk_normalize(X2, axis=1)]).tocsr()
                self.X = sk_normalize(self.X, axis=1, copy=False)

                if os.path.exists(f'{self.data_dir}/Y.trn.npz'):
                    self.Y = smat.load_npz(f'{self.data_dir}/Y.trn.npz')
                else:
                    raise RuntimeError(f'Can\'t find any Training Data, please check if {self.data_dir}/Y.trn.npz exists')
                
                label_feat = LabelEmbeddingFactory.create(self.Y, self.X, method = None, dtype=self.X.dtype)

                if os.path.exists(f'{self.output_dir}/indexer/code.npz'):
                    self.C = Indexer.load_indexed_code(f'{self.output_dir}/indexer/code.npz', label_feat)
                else:
                    raise RuntimeError(f'Can\'t find any Training Data, please check if {self.output_dir}/indexer/code.npz exists')
                
                if os.path.exists(f'{self.matcher_dir}/C_trn_pred.npz'):
                    self.Z_pred = smat.load_npz(f'{self.matcher_dir}/C_trn_pred.npz')
                else:
                    raise RuntimeError(f'Can\'t find any Training Data, please check if {self.output_dir}/indexer/code.npz exists')
                
            self.logger.info('Finish Loading Data for training')
            
            # else:

            #     self.logger.info('Data for training was loaded')
                
            
        elif phase == 'predict':

            if (self.Xt is None or self.Yt is None or self.csr_codes is None):

                if os.path.exists(f'{self.data_dir}/X.tst.npz'):
                    X1 = HierarchicalMLModel.load_feature_matrix(f'{self.data_dir}/X.tst.npz')
                else:
                    raise RuntimeError(f'Can\'t find any Training Data, please check if {self.data_dir}/X.tst.npz exists')
                
                if os.path.exists(f'{self.matcher_dir}/tst_embeddings.npy'):
                    X2 = HierarchicalMLModel.load_feature_matrix(f'{self.matcher_dir}/tst_embeddings.npy')
                else:
                    raise RuntimeError(f'Can\'t find any Training Data, please check if {self.matcher_dir}/tst_embeddings.npy exists')

                self.Xt = smat.hstack([sk_normalize(X1, axis=1), sk_normalize(X2, axis=1)]).tocsr()

                if os.path.exists(f'{self.matcher_dir}/C_tst_pred.npz'):
                    self.csr_codes = smat.load_npz(f'{self.matcher_dir}/C_tst_pred.npz')
                else:
                    raise RuntimeError(f'Can\'t find any Training Data, please check if {self.matcher_dir}/C_tst_pred.npz exists')
                
                if os.path.exists(f'{self.data_dir}/Y.tst.npz'):
                    self.Yt = smat.load_npz(f'{self.data_dir}/Y.tst.npz')
                else:
                    raise RuntimeError(f'Can\'t find any Training Data, please check if {self.data_dir}/Y.tst.npz exists')
            
            self.logger.info('Finish Loading Data for prediction')
            
            # else:

            #     self.logger.info('Data for prediction was loaded')
        
        elif phase == 'eval':

            if (self.Yt is None):

                if os.path.exists(f'{self.data_dir}/Y.tst.npz'):
                    self.Yt = smat.load_npz(f'{self.data_dir}/Y.tst.npz')
                else:
                    raise RuntimeError(f'Can\'t find any Training Data, please check if {self.data_dir}/Y.tst.npz exists')
                
            self.logger.info('Finish Loading Data for evaluation')
            
            # else:

            #     self.logger.info('Data for evaluation was loaded')


        else:
            raise RuntimeError(f'Unkown phase {phase}')

    def __predict(self):
                
        self.load_data('predict')
        config = self.config['predict']

        model = LinearModel.load(self.ranker_dir)
        model = model[-1]
        cond_prob = PostProcessor.get(config['transform'])
        Yt_pred = model.predict(self.Xt, csr_codes=self.csr_codes, beam_size=config['beam_size'], only_topk=config['only_topk'], cond_prob=cond_prob,)

        metric = Metrics.generate(self.Yt, Yt_pred, topk=10)
        self.logger.log("==== tst_set evaluation ====")
        self.logger.log(metric)

        smat.save_npz(f'{self.ranker_dir}/tst.pred.npz', Yt_pred)

    def eval(self):

        self.load_data('eval')

        pred_paths = [f'{self.output_dir}/ranker/{model_name}/tst.pred.npz' for model_name in self.model_name_arr]

        Y_pred_list = []
        for pred_path in pred_paths:
            if not os.path.exists(pred_path):
                raise Warning("pred_path does not exists: {}".format(pred_path))
            else:
                Y_pred = smat.load_npz(pred_path)
                Y_pred.data = rf_linear.Transform.sigmoid(Y_pred.data)

                Y_pred_list += [Y_pred]
                self.logger.log("==== Evaluation on {}".format(pred_path))
                self.logger.log(rf_linear.Metrics.generate(self.Yt, Y_pred))

        if len(Y_pred_list) > 1:
            self.logger.log("==== Evaluations of Ensembles of All Predictions ====")
            for ens in [
                rf_linear.CsrEnsembler.average,
                rf_linear.CsrEnsembler.rank_average,
                rf_linear.CsrEnsembler.round_robin,
            ]:
                self.logger.log("ens: {}".format(ens.__name__))
                self.logger.log(rf_linear.Metrics.generate(self.Yt, ens(*Y_pred_list)))

