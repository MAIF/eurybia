"""
Detection Drift Model Module
"""

catboost_hyperparameter_type = {
    "max_depth": ["<class 'int'>"],
    "l2_leaf_reg": ["<class 'float'>", "<class 'int'>"],
    "learning_rate": ["<class 'float'>"],
    "iterations": ["<class 'int'>"],
    "use_best_model": ["<class 'bool'>"],
    "custom_loss": ["<class 'str'>", "<class 'bool'>"],
    "loss_function": ["<class 'str'>", "<class 'object'>"],
    "eval_metric": ["<class 'str'>", "<class 'object'>"],
}

catboost_hyperparameter_init = {
    "max_depth": 8,
    "l2_leaf_reg": 19,
    "learning_rate": 0.166905,
    "iterations": 250,
    "use_best_model": True,
    "custom_loss": ["Accuracy", "AUC", "Logloss"],
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
}
