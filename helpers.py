

def save_model(models, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(models, handle)
        
def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)
    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y

def lgb_trainer(X, y, params, n_folds):
    skf = StratifiedKFold(n_splits=n_folds)
    models = []
    for train_idx, test_idx in skf.split(X.values, y.values):
            gc.collect()
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_tr, y_tr = augment(X_train.values, y_train.values)
            X_tr = pd.DataFrame(X_tr)
            trn_data = lgb.Dataset(X_tr, label=y_tr)
            test_data = lgb.Dataset(X.values[test_idx], label=y.values[test_idx])
            model_lgb     = lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, test_data], verbose_eval=5000, early_stopping_rounds = 4000)
            models.append(model_lgb)
            auc = roc_auc_score(y.values[test_idx], model_lgb.predict(X.values[test_idx]))
    return models

def lgb_trainer_no_aug(X, y, params, n_folds):
    skf = StratifiedKFold(n_splits=n_folds)
    models = []
    for train_idx, test_idx in skf.split(X.values, y.values):
            gc.collect()
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_tr, y_tr = X_train.values, y_train.values
            X_tr = pd.DataFrame(X_tr)
            trn_data = lgb.Dataset(X_tr, label=y_tr)
            test_data = lgb.Dataset(X.values[test_idx], label=y.values[test_idx])
            model_lgb     = lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, test_data], verbose_eval=5000, early_stopping_rounds = 4000)
            models.append(model_lgb)
            auc = roc_auc_score(y.values[test_idx], model_lgb.predict(X.values[test_idx]))
    return models

def test(X, y, models):
    preds = pd.DataFrame({})
    for i, model in enumerate(models):
        preds[str(i)] = model.predict(X)
        print(f"Fold: {i} \t Score: {roc_auc_score(y, preds[str(i)].values)}")
    averaged_preds = preds.mean(axis=1)
    print(f"Score: {roc_auc_score(y, averaged_preds)}")
    return averaged_preds, preds