# Testing configurations
config = {}

training_opt = {}
training_opt['num_classes'] = 1000
training_opt['batch_size'] = 128
training_opt['num_workers'] = 8
training_opt['num_epochs'] = 60
training_opt['display_step'] = 10
training_opt['feature_dim'] = 512
training_opt['open_threshold'] = 0.1
training_opt['sampler'] = {'type': 'ClassAwareSampler', 'def_file': 'ClassAwareSampler',
                           'num_samples_cls': 4}
training_opt['scheduler_params'] = {'step_size': 20, 'gamma': 0.1}
config['training_opt'] = training_opt

networks = {}
feature_param = {'use_modulatedatt': True, 'use_fc': True, 'dropout': None,
                 'stage1_weights': True, 'dataset': training_opt['dataset']}
feature_optim_param = {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0.0005}
networks['feat_model'] = {'def_file': 'ResNet10Feature',
                          'params': feature_param,
                          'optim_params': feature_optim_param,
                          'fix': False}
classifier_param = {'in_dim': training_opt['feature_dim'], 'num_classes': training_opt['num_classes'],
                    'stage1_weights': True, 'dataset': training_opt['dataset']}
classifier_optim_param = {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 0.0005}
networks['classifier'] = {'def_file': 'MetaEmbeddingClassifier',
                          'params': classifier_param,
                          'optim_params': classifier_optim_param}
config['networks'] = networks

criterions = {}
perf_loss_param = {}
criterions['PerformanceLoss'] = {'def_file': 'SoftmaxLoss', 'loss_params': perf_loss_param,
                                 'optim_params': None, 'weight': 1.0}
feat_loss_param = {'feat_dim': training_opt['feature_dim'], 'num_classes': training_opt['num_classes']}
feat_loss_optim_param = {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0.0005}
criterions['FeatureLoss'] = {'def_file': 'DiscCentroidsLoss', 'loss_params': feat_loss_param,
                             'optim_params': feat_loss_optim_param, 'weight': 0.01}
config['criterions'] = criterions

memory = {}
memory['centroids'] = True
memory['init_centroids'] = True
config['memory'] = memory
