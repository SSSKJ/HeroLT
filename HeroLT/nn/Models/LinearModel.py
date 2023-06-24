from ..xbert.rf_linear import MLProblem, HierarchicalMLModel

solver_dict = {
    #'L2R_LR':0,
    "L2R_L2LOSS_SVC_DUAL": 1,
    #'L2R_L2LOSS_SVC':2,
    "L2R_L1LOSS_SVC_DUAL": 3,
    #'MCSVM_CS':4,
    "L1R_L2LOSS_SVC": 5,
    #'L1R_LR':6,
    "L2R_LR_DUAL": 7,
}

class LinearModel(object):
    def __init__(self, model=None):
        self.model = model

    def __getitem__(self, key):
        return LinearModel(self.model[key])

    def __add__(self, other):
        return LinearModel(self.model + other.model, self.bias)

    def save(self, model_folder):
        self.model.save(model_folder)

    @classmethod
    def load(cls, model_folder):
        return cls(HierarchicalMLModel.load(model_folder))

    @classmethod
    def train(
        cls,
        X,
        Y,
        C,
        mode="full-model",
        shallow=False,
        solver_type=solver_dict["L2R_L2LOSS_SVC_DUAL"],
        Cp=1.0,
        Cn=1.0,
        threshold=0.1,
        max_iter=100,
        threads=-1,
        bias=-1.0,
        Z_pred=None,
        negative_sampling_scheme=None,
    ):
        if mode in ["full-model", "matcher"]:
            if mode == "full-model":
                prob = MLProblem(X, Y, C, Z_pred=Z_pred, negative_sampling_scheme=negative_sampling_scheme,)
            elif mode == "matcher":
                assert C is not None
                Y = Y.dot(C)
                prob = MLProblem(X, Y, C=None)

            hierarchical = True
            min_labels = 2
            if shallow:
                if prob.C is None:
                    min_labels = prob.Y.shape[1]
                else:
                    min_labels = prob.C.shape[1]
        elif mode == "ranker":
            assert C is not None
            prob = MLProblem(X, Y, C, Z_pred=Z_pred, negative_sampling_scheme=negative_sampling_scheme,)
            hierarchical = False
            min_labels = 2

        model = HierarchicalMLModel.train(
            prob,
            hierarchical=hierarchical,
            min_labels=min_labels,
            solver_type=solver_type,
            Cp=Cp,
            Cn=Cn,
            threshold=threshold,
            threads=threads,
            bias=bias,
            max_iter=max_iter,
        )
        return cls(model)

    def predict(self, X, csr_codes=None, beam_size=10, only_topk=10, cond_prob=True):
        pred_csr = self.model.predict(X, only_topk=only_topk, csr_codes=csr_codes, beam_size=beam_size, cond_prob=cond_prob,)
        return pred_csr