import numpy as np
from backbone import LGBM


def lgbm_model_search(X_train, Y_train, X_val, Y_val):
    # grid search on parameters space

    # grid search on lr and base
    auc0, lr0, base0, pred0, model0 = 0, 0, 0, []
    for lr in np.arange(0.1, 0.9, 0.01):
        for base in np.arange(0.9, 0.99, 0.01):
            model, pred, auc = LGBM(X_train, Y_train, X_val, Y_val, lr = lr, base = base)
            if auc>auc0:
                auc0, lr0, base0, pred0 = auc, lr, base, pred

    auc1, lr1, base1, pred1 = 0, 0, 0, []
    for lr in np.arange(lr0-0.1, lr0+0.1, 0.001):
        for base in np.arange(base0-0.1, base0+0.1, 0.01):
            model, pred, auc = LGBM(X_train, Y_train, X_val, Y_val, lr = lr, base = base)
            if auc>auc1:
                auc1, lr1, base1, pred1, model1 = auc, lr, base, pred, model

    auc2, lr2, base2, pred2 = 0, 0, 0, []
    for lr in np.arange(lr1-0.01, lr1+0.01, 0.0001):
        for base in np.arange(base1-0.01, base+0.01, 0.01):
            model, pred, auc = LGBM(X_train, Y_train, X_val, Y_val, lr = lr, base = base)
            if auc>auc2:
                auc2, lr2, base2, pred2, model2 = auc, lr, base, pred, model

    print(f"Best model: AUC: {auc0}, LR: {lr0}, Base: {base0}" )

    return model2

