from imblearn.over_sampling import SMOTE


#SMOTE过采样
def smote(x,y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(x, y)
    return X_res, y_res



