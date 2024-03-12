import lightgbm as lgb


#### lgbm
def LGBM(X_train, Y_train, X_val, Y_val, lr = 0.1, base = 0.9):
    params = {
        "boosting_type": 'gbdt',
        "n_estimators": 100,
        "num_leaves":100,
        "max_depth": 50,
        "objective" : "binary",
        'learning_rate': lr,
        "metric" : "auc",
        "num_leaves" : 100,
        "bagging_fraction" : 0.9,
        "feature_fraction" : 0.9,
        "bagging_seed" : 42,
        "verbose" : -1,
        "seed": 42,
        "early_stopping_rounds":2
    }
    eval_set = {}
    model = lgb.LGBMClassifier(**params).fit(X_train, Y_train.ravel(), eval_metric=['auc'],
                      eval_set=[(X_train, Y_train),(X_val, Y_val)],
                      callbacks=[lgb.record_evaluation(eval_set),
                      lgb.reset_parameter(learning_rate=lambda x: lr*base**x)])

    pred = model.predict(X_val)
    auc = model.score(X_val, Y_val)
    print(lr, base, auc)

    return model, pred, auc