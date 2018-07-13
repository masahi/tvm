"""Tuner that uses xgboost as cost model"""

from .model_based_tuner import ModelBasedTuner, ModelOptimizer
from .xgboost_cost_model import XGBoostCostModel
from .sa_model_optimizer import SimulatedAnnealingOptimizer

class XGBTuner(ModelBasedTuner):
    """Tuner that uses xgboost as cost model

    Parameters
    ----------
    task: Task
        The tuning task
    plan_size: int
        The size of a plan. After `plan_size` trials, the tuner will refit a new cost model
        and do planing for the next `plan_size` trials.
    feature_type: str, optional
        If is 'itervar', use features extracted from IterVar (loop variable).
        If is 'knob', use flatten ConfigEntity directly.
        If is 'curve', use sampled curve feature (relation feature).

        Note on choosing feature type:
        For single task tuning, 'itervar' and 'knob' is good.
                                'itervar' is more accurate but 'knob' is much faster.
        For cross-shape tuning (e.g. many convolutions with different shapes),
                               'itervar' and 'curve' has better transferability,
                               'knob' is faster.
        For cross-device or cross-operator tuning, you can use 'curve' only.
    loss_type: str
        If is 'reg', use regression loss to train cost model.
                     The cost model predicts the normalized flops.
        If is 'rank', use pairwise rank loss to train cost model.
                     The cost model predicts relative rank score.
    num_threads: int, optional
        The number of threads.
    optimizer: str or ModelOptimizer, optional
        If is 'sa', use a default simulated annealing optimizer.
        Otherwise it should be a ModelOptimizer object.
    diversity_filter_ratio: int or float, optional
        If is not None, the tuner will first select
        top-(plan_size * diversity_filter_ratio) candidates according to the cost model
        and then pick batch_size of them according to the diversity metric.
    """
    def __init__(self, task, plan_size=32,
                 feature_type='itervar', loss_type='rank', num_threads=None,
                 optimizer='sa', diversity_filter_ratio=None):
        cost_model = XGBoostCostModel(task,
                                      feature_type=feature_type,
                                      loss_type=loss_type,
                                      num_threads=num_threads)
        if optimizer == 'sa':
            optimizer = SimulatedAnnealingOptimizer(task)
        else:
            assert isinstance(optimizer, ModelOptimizer), "Optimizer must be " \
                                                          "a supported name string" \
                                                          "or a ModelOptimizer object."

        super(XGBTuner, self).__init__(task, cost_model, optimizer,
                                       plan_size, diversity_filter_ratio)
