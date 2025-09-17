from py_boost.multioutput.sketching import *
PYBOOST_SAMPLING_METHODS = dict(filter=FilterSketch,
                                #
                                svd=SVDSketch,
                                # using approximation to truncate hessian and grad dims
                                random_sampling=RandomSamplingSketch,
                                # using "probabilities" obtain from columns
                                random_projection=RandomProjectionSketch
                                # using random matrix from Normal distribution/ Radamacher distribution
                                )

N_TREES = 10000
VERBOSE = 100
DEFAULT_SKETCH_METHOD = 'random_projection'
BOOSTING_MODEL_PARAMS = dict(loss='crossentropy', ntrees=N_TREES, lr=0.03, verbose=VERBOSE,
                             es=300, lambda_l2=1, gd_steps=1,
                             subsample=1, colsample=1,
                             min_data_in_leaf=10, max_bin=256,
                             max_depth=6, multioutput_sketch=PYBOOST_SAMPLING_METHODS[DEFAULT_SKETCH_METHOD](5))
