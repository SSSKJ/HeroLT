# from . import BaseModel

# import os
# import sys
# import json
# import logging
# import time
# from ..pecos.core import XLINEAR_SOLVERS
# from ..pecos.utils import cli
# from ..pecos.utils import smat_util, logging_util
# from ..pecos.utils.cluster_util import ClusterChain
# from ..pecos.xmc import Indexer, LabelEmbeddingFactory, PostProcessor
# from ..pecos.xmc.base import HierarchicalKMeans
# from ..pecos.xmc.xlinear.model import XLinearModel
# from sklearn.preprocessing import normalize


# LOGGER = logging.getLogger()
# LOGGER.setLevel(logging.DEBUG)

# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# handler.setFormatter(formatter)
# LOGGER.addHandler(handler)

# class XRLinear(BaseModel):


#     def __init__(
#             self,
#             dataset: str,
#             base_dir: str = '../../',
#             ) -> None:
        
#         super().__init__(
#             model_name = 'XRLinear',
#             dataset_name = dataset,
#             base_dir = base_dir)
        
#         super().__load_config()
#         self.X = None
#         self.Y = None
#         self.Xt = None

#     def train(self):

#         self.ns_scheme= 'man'  # man, tfn+man
#         self.data_dir=f'{self.base_dir}/data/xmc-base/{self.dataset_name}'
#         # output_dir=./exp_v2

#         # EXP-2
#         self.nr_splits=32
#         seed_arr=[0, 1, 2]
#         beam_arr=[10, 20, 50]

#         for beam_size in beam_arr:

#             pred_npz_string=""
#             pred_tag_string=""
            
#             for seed in seed_arr:

#                 pred_npz=f'{self.output_path}/Yp.tst.b-{beam_size}.npz'
#                 pred_tag=f'{self.ns_scheme}_seed-${seed}'
#                 self.__train_model()
#                 self.__predict()

#                 pred_npz_string="${pred_npz_string} ${pred_npz}"
#                 pred_tag_string="${pred_tag_string} ${pred_tag}"

#             ens_method_arr=['average', 'rank_average', 'softmax_average', 'sigmoid_average']
#             for ens_method in ens_method_arr:
#                 self.__ensemble_evaluate(ens_method)

#     ## todo
#     def load_data(self, phase):
        
#         if phase == 'train' and (self.X is None or self.Y is None):
#             LOGGER.info("| loading data begin...")
#             start_time = time.time()
#             self.X = XLinearModel.load_feature_matrix(args.inst_path) ## todo: change into right name
#             self.X = normalize(self.X, axis=1, norm="l2")
#             self.Y = XLinearModel.load_label_matrix(args.label_path, for_training=True) ## todo: change into right name
#             run_time_io = time.time() - start_time
#             LOGGER.info("| loading data finsihed | time(s) {:9.4f}".format(run_time_io))

#         elif phase == 'predict' and self.Xt is None:

#             # Load data
#             LOGGER.info("| loading data begin...")
#             start_time = time.time()
#             self.Xt = XLinearModel.load_feature_matrix(args.inst_path)
#             self.Xt = normalize(self.Xt, axis=1, norm="l2")
#             run_time_data = time.time() - start_time
#             LOGGER.info("| loading data finsihed | time(s) {:9.4f}".format(run_time_data))
        
#     def __train_model(self):
        
#         """Train and Save xr-linear model

#         Args:
#             args (argparse.Namespace): Command line arguments parsed by `parser.parse_args()`
#         """
#         params = dict()
#         # if args.generate_params_skeleton:
#         #     params["train_params"] = XLinearModel.TrainParams.from_dict({}, recursive=True).to_dict()
#         #     params["pred_params"] = XLinearModel.PredParams.from_dict({}, recursive=True).to_dict()
#         #     params["indexer_params"] = HierarchicalKMeans.TrainParams.from_dict(
#         #         {}, recursive=True
#         #     ).to_dict()
#         #     print(f"{json.dumps(params, indent=True)}")
#         #     return

#         # if args.params_path:
#         #     with open(args.params_path, "r") as fin:
#         #         params = json.load(fin)

#         train_params = params.get("train_params", None)
#         pred_params = params.get("pred_params", None)
#         indexer_params = params.get("indexer_params", None)

#         if train_params is not None:
#             train_params = XLinearModel.TrainParams.from_dict(train_params)
#         else:
#             train_params = XLinearModel.TrainParams.from_dict(
#                 {k: v for k, v in vars(self.config).items() if v is not None},
#                 recursive=True,
#             )

#         if pred_params is not None:
#             pred_params = XLinearModel.PredParams.from_dict(pred_params)
#         else:
#             pred_params = XLinearModel.PredParams.from_dict(
#                 {k: v for k, v in vars(self.config).items() if v is not None},
#                 recursive=True,
#             )

#         if indexer_params is not None:
#             indexer_params = HierarchicalKMeans.TrainParams.from_dict(indexer_params)
#         else:
#             indexer_params = HierarchicalKMeans.TrainParams.from_dict(
#                 {k: v for k, v in vars(self.config).items() if v is not None},
#                 recursive=True,
#             )
#         if self.config.seed:
#             indexer_params.seed = self.config.seed

#         # if not os.path.exists(args.model_folder):
#         #     os.makedirs(args.model_folder)

#         self.load_data('train')

#         LOGGER.info("| building HLT...")
#         start_time = time.time()
#         if args.code_path:
#             cluster_chain = ClusterChain.load(args.code_path)
#         else:
#             if args.label_feat_path:
#                 label_feat = XLinearModel.load_feature_matrix(args.label_feat_path)
#             else:
#                 label_feat = LabelEmbeddingFactory.create(self.Y, self.X, method="pifa")

#             cluster_chain = Indexer.gen(label_feat, train_params=indexer_params)
#         run_time_hlt = time.time() - start_time
#         LOGGER.info("| building HLT finsihed | time(s) {:9.4f}".format(run_time_hlt))

#         # load label importance matrix if given
#         if args.usn_label_path:
#             usn_label_mat = smat_util.load_matrix(args.usn_label_path)
#         else:
#             usn_label_mat = None
#         # load user supplied matching matrix if given
#         if args.usn_match_path:
#             usn_match_mat = smat_util.load_matrix(args.usn_match_path)
#         else:
#             usn_match_mat = None
#         usn_match_dict = {0: usn_label_mat, 1: usn_match_mat}

#         # load relevance matrix for cost-sensitive learning
#         if args.rel_path:
#             R = smat_util.load_matrix(args.rel_path)
#         else:
#             R = None

#         pred_kwargs = {}
#         for kw in ["beam_size", "only_topk", "post_processor"]:
#             if getattr(args, kw, None) is not None:
#                 pred_kwargs[kw] = getattr(args, kw)

#         LOGGER.info("| training XR-Linear...")
#         start_time = time.time()
#         xlm = XLinearModel.train(
#             self.X,
#             self.Y,
#             C=cluster_chain,
#             R=R,
#             user_supplied_negatives=usn_match_dict,
#             train_params=train_params,
#             pred_params=pred_params,
#             pred_kwargs=pred_kwargs,
#         )
#         run_time_xrl = time.time() - start_time
#         LOGGER.info("| training XR_Linear finsihed | time(s) {:9.4f}".format(run_time_xrl))

#         xlm.save(args.model_folder)
#         LOGGER.info("| Finished with run_time(s) | total {:9.4f} hlt {:9.4f} xrl {:9.4f}".format(
#             run_time_hlt + run_time_xrl,
#             run_time_hlt,
#             run_time_xrl,
#             )
#         )

#     def __predict(self):

#         """Predict and Evaluate for xlinear model

#         Args:
#             args (argparse.Namespace): Command line arguments parsed by `parser.parse_args()`
#         """

#         self.load_data(phase = 'predict')

#         LOGGER.info("| loading model begin...")
#         start_time = time.time()
#         if args.selected_output is not None:
#             # Selected Output
#             selected_outputs_csr = XLinearModel.load_feature_matrix(args.selected_output)
#             xlinear_model = XLinearModel.load(
#                 args.model_folder, is_predict_only=True, weight_matrix_type="CSC"
#             )
#         else:
#             # TopK
#             selected_outputs_csr = None
#             xlinear_model = XLinearModel.load(args.model_folder, is_predict_only=True)
#         run_time_io = time.time() - start_time
#         LOGGER.info("| loading model finsihed | time(s) {:9.4f}".format(run_time_io))


#         # Model Predicting
#         LOGGER.info("| inference model begin...")
#         start_time = time.time()
#         Yt_pred = xlinear_model.predict(
#             Xt,
#             selected_outputs_csr=selected_outputs_csr,
#             only_topk=args.only_topk,
#             beam_size=args.beam_size,
#             post_processor=args.post_processor,
#             threads=args.threads,
#             max_pred_chunk=args.max_pred_chunk,
#         )
#         run_time_pred = time.time() - start_time
#         LOGGER.info("| inference model finsihed | time(s) {:9.4f} latency(ms/q) {:9.4f}".format(
#             run_time_pred,
#             run_time_pred / Xt.shape[0] * 1000,
#             )
#         )

#         # Save prediction
#         if args.save_pred_path:
#             smat_util.save_matrix(args.save_pred_path, Yt_pred)

#         # Evaluate
#         if args.label_path:
#             Yt = XLinearModel.load_label_matrix(args.label_path)
#             metric = smat_util.Metrics.generate(Yt, Yt_pred, topk=10)
#             print("==== evaluation results ====")
#             print(metric)

#     def __ensemble_evaluate(self, ens_method):

#         """ Evaluate xlinear predictions """
#         assert len(args.tags) == len(args.pred_path)
#         Y_true = smat_util.sorted_csr(smat_util.load_matrix(args.truth_path).tocsr())
#         Y_pred = [smat_util.sorted_csr(smat_util.load_matrix(pp).tocsr()) for pp in args.pred_path]
#         print("==== evaluation results ====")
#         smat_util.CsrEnsembler.print_ens(Y_true, Y_pred, args.tags, ens_method=ens_method)