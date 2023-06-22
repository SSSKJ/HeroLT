from . import BaseModel
from ...utils.logger import get_logger
from ..pecos.core import XLINEAR_SOLVERS
from ..pecos.utils import cli
from ..pecos.utils import smat_util, logging_util
from ..pecos.utils.cluster_util import ClusterChain
from ..pecos.xmc import Indexer, LabelEmbeddingFactory, PostProcessor
from ..pecos.xmc.base import HierarchicalKMeans
from ..pecos.xmc.xlinear.model import XLinearModel

from sklearn.preprocessing import normalize

import time
import os

## set logger
class XRLinear(BaseModel):


    def __init__(
            self,
            dataset: str,
            base_dir: str = '../../',
            ) -> None:
        
        super().__init__(
            model_name = 'XRLinear',
            dataset_name = dataset,
            base_dir = base_dir)
        
        super().load_config()

        self.X = None
        self.Y = None
        self.Xt = None
        self.Yt = None
        self.xlinear_model = None

        self.ns_scheme = self.config['ns_scheme']
        self.data_dir=f'{self.base_dir}/data/xmc-base/{self.dataset_name}'
        self.seed_arr=[0, 1, 2]
        self.beam_arr=[10, 20, 50]
        self.ens_method_arr=['average', 'rank_average', 'softmax_average', 'sigmoid_average']

        self.logger = get_logger(self.base_dir, f'{self.model_name}_{self.dataset_name}.log')

    def train(self):

        # EXP-2

        for beam_size in self.beam_arr:
            
            for seed in self.seed_arr:

                self.config['seed'] = seed
                self.config['beam_size'] = beam_size
                self.__train_model(seed, beam_size)
                self.__predict(beam_size)
            
            for ens_method in self.ens_method_arr:
                self.__ensemble_evaluate(ens_method)

    ## todo
    def load_data(self, phase):
        
        if phase == 'train' and (self.X is None or self.Y is None):
            self.logger.info("| loading data begin...")
            start_time = time.time()
            if os.path.exists(f'{self.data_dir}/tfidf-attnxml/X.trn.npz'):
                self.X = XLinearModel.load_feature_matrix(f'{self.data_dir}/tfidf-attnxml/X.trn.npz') ## todo: change into right name
            else:
                raise RuntimeError(f'Can\'t find any Training Data, please check if {self.data_dir}/tfidf-attnxml/X.trn.npz exists')
            self.X = normalize(self.X, axis=1, norm="l2")

            if os.path.exists(f'{self.data_dir}/Y.trn.npz'):
                self.Y = XLinearModel.load_label_matrix(f'{self.data_dir}/Y.trn.npz', for_training=True) ## todo: change into right name
            else:
                raise RuntimeError(f'Can\'t find any Training Data, please check if {self.data_dir}/Y.trn.npz exists')
            
            run_time_io = time.time() - start_time
            self.logger.info("| loading data finsihed | time(s) {:9.4f}".format(run_time_io))

        elif phase == 'predict' and (self.Xt is None or self.Yt is None):

            # Load data
            self.logger.info("| loading data begin...")
            start_time = time.time()
            if os.path.exists(f'{self.data_dir}/tfidf-attnxml/X.tst.npz'):
                self.Xt = XLinearModel.load_feature_matrix(f'{self.data_dir}/tfidf-attnxml/X.tst.npz')
            else:
                raise RuntimeError(f'Can\'t find any Training Data, please check if {self.data_dir}/tfidf-attnxml/X.tst.npz exists')
            
            self.Xt = normalize(self.Xt, axis=1, norm="l2")

            if os.path.exists(f'{self.data_dir}/Y.tst.npz'):
                self.Yt = XLinearModel.load_label_matrix(f'{self.data_dir}/Y.tst.npz')
            else:
                raise RuntimeError(f'Can\'t find any Training Data, please check if {self.data_dir}/Y.tst.npz exists')
            
            run_time_data = time.time() - start_time
            self.logger.info("| loading data finsihed | time(s) {:9.4f}".format(run_time_data))

    def load_pretrained_model(self):
        
        self.xlinear_model = XLinearModel.load(self.output_path, is_predict_only=True)
        
        
    def __train_model(self, seed, beam_size):

        self.train_params = XLinearModel.TrainParams.from_dict(
            {k: v for k, v in self.config.items() if v is not None},
            recursive=True,
        )

        self.pred_params = XLinearModel.PredParams.from_dict(
            {k: v for k, v in self.config.items() if v is not None},
            recursive=True,
        )

        self.indexer_params = HierarchicalKMeans.TrainParams.from_dict(
            {k: v for k, v in self.config.items() if v is not None},
            recursive=True,
        )

        self.indexer_params.seed = seed

        self.load_data('train')

        self.logger.info("| building HLT...")
        start_time = time.time()
        label_feat = LabelEmbeddingFactory.create(self.Y, self.X, method="pifa")
        cluster_chain = Indexer.gen(label_feat, train_params=self.indexer_params)

        run_time_hlt = time.time() - start_time
        self.logger.info("| building HLT finsihed | time(s) {:9.4f}".format(run_time_hlt))

        # load label importance matrix if given
        usn_label_mat = None
        # load user supplied matching matrix if given
        usn_match_mat = None
        usn_match_dict = {0: usn_label_mat, 1: usn_match_mat}

        # load relevance matrix for cost-sensitive learning
        R = None

        pred_kwargs = {}
        pred_kwargs[beam_size] = beam_size

        self.logger.info("| training XR-Linear...")
        start_time = time.time()
        self.xlinear_model = XLinearModel.train(
            self.X,
            self.Y,
            C=cluster_chain,
            R=R,
            user_supplied_negatives=usn_match_dict,
            train_params=self.train_params,
            pred_params=self.pred_params,
            pred_kwargs=pred_kwargs,
        ) # todo: logger
        run_time_xrl = time.time() - start_time
        self.logger.info("| training XR_Linear finsihed | time(s) {:9.4f}".format(run_time_xrl))

        self.xlinear_model.save(self.output_path)
        self.logger.info("| Finished with run_time(s) | total {:9.4f} hlt {:9.4f} xrl {:9.4f}".format(
            run_time_hlt + run_time_xrl,
            run_time_hlt,
            run_time_xrl,
            )
        )

    def __predict(self, beam_size):

        """Predict and Evaluate for xlinear model

        Args:
            args (argparse.Namespace): Command line arguments parsed by `parser.parse_args()`
        """

        self.load_data(phase = 'predict')

        self.logger.info("| loading model begin...")
        start_time = time.time()
        selected_outputs_csr = None

        if self.xlinear_model is None:
            self.load_pretrained_model()

        run_time_io = time.time() - start_time
        self.logger.info("| loading model finsihed | time(s) {:9.4f}".format(run_time_io))


        # Model Predicting
        self.logger.info("| inference model begin...")
        start_time = time.time()
        Yt_pred = self.xlinear_model.predict(
            self.Xt,
            selected_outputs_csr=selected_outputs_csr,
            threads=self.config['threads'],
            max_pred_chunk=self.config['max_pred_chunk'],
        )
        run_time_pred = time.time() - start_time
        self.logger.info("| inference model finsihed | time(s) {:9.4f} latency(ms/q) {:9.4f}".format(
            run_time_pred,
            run_time_pred / self.Xt.shape[0] * 1000,
            )
        )

        # Save prediction
        smat_util.save_matrix(f'{self.output_path}/Yp.tst.b-{beam_size}.npz', Yt_pred)

        # Evaluate
        metric = smat_util.Metrics.generate(self.Yt, Yt_pred, topk=10)
        self.logger.log("==== evaluation results ====")
        self.logger.log(metric)

    def __ensemble_evaluate(self, ens_method):

        """ Evaluate xlinear predictions """
        Y_true = smat_util.sorted_csr(smat_util.load_matrix(f'{self.data_dir}/Y.tst.npz').tocsr())
        Y_pred = [smat_util.sorted_csr(smat_util.load_matrix(pp).tocsr()) for pp in [f'{self.output_path}/Yp.tst.b-{beam_size}.npz' for beam_size in self.beam_arr]]
        print("==== evaluation results ====")
        smat_util.CsrEnsembler.print_ens(Y_true, Y_pred, [f'{self.ns_scheme}_seed-{seed}' for seed in self.seed_arr], ens_method=ens_method)

        