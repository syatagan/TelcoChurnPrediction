#########################################################
# Business Problem
# A machine learning model is expected to be developed that can predict customers will churn the company or not.
#########################################################
#
# Dataset Story
# Telco churn data includes information about a fictitious telecom company that, in the third quarter,
# provided home phone and Internet services to 7043 customers in California.
# Shows which customers have left, stayed or signed up for their service.
"""
CustomerId      Müşteri İd’si
Gender          Cinsiyet
SeniorCitizen   Müşterinin yaşlı olup olmadığı (1, 0)
Partner         Müşterinin bir ortağı olup olmadığı (Evet, Hayır)
Dependents      Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır
tenure          Müşterinin şirkette kaldığı ay sayısı
PhoneService    Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
MultipleLines   Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
InternetService Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
OnlineSecurity  Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
OnlineBackup    Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
DeviceProtection Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
TechSupport     Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
StreamingTV     Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
StreamingMovies Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
Contract        Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
PaperlessBilling Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
PaymentMethod   Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
MonthlyCharges  Müşteriden aylık olarak tahsil edilen tutar
TotalCharges    Müşteriden tahsil edilen toplam tutar
Churn           Müşterinin kullanıp kullanmadığı (Evet veya Hayır
"""
# !pip install catboost
# !pip install xgboost
# !pip install lightgbm

# imports
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from Src.utils import grab_col_names,replace_with_thresholds, check_outlier, check_MissingValue
from Src.utils import check_df,cat_summary,num_summary,target_summary_with_num,target_summary_with_cat
from Src.utils import plot_importance, outlier_analyser, missing_value_analyser, one_hot_encoder
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score, confusion_matrix
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import shap
shap.initjs()

# Settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)

# read data
df = pd.read_csv("Datasets/Telco-Customer-Churn.csv")
df_ = df.copy()

df = df_
df.columns = [col.lower() for col in df.columns]
##################################################################
# DUTY 1 : Explonatory Data Analysis
##################################################################
check_df(df,xplot=True)

# Stage 2 : Check for Variable Types
df["seniorcitizen"] = df["seniorcitizen"].astype(object)
df["totalcharges"].value_counts().sort_values()   # 11 tane " " verisi var.
df["totalcharges"] = df["totalcharges"].replace(" ",0).astype(float)
df.dtypes

# Stage 1 : Grab numeric and categorigal variables
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Stage 3 : Observe the distribution of numerical and categorical variables in the data
for col in cat_cols:
    cat_summary(df, col, True)
for col in num_cols:
    num_summary(df, col, False)

# Stage 4 : Do the target variable analysis with categorical variables
for col in cat_cols:
    target_summary_with_cat(df, "churn", col )

for col in num_cols:
    target_summary_with_num(df, "churn", col )

# Stage 5 : Check for outlier values.
outlier_analyser(df, num_cols)

# Stage 6 : Check for Missing Values , "
missing_value_analyser(df, df.columns)

################################################################
# DUTY 2 : Feature Engineering
#################################################################
# Stage 1 : Take necessary actions for missing and outlier values.
for col in cat_cols:
    if len(df.loc[df[col] == " "]) > 0:
        print(col)

for col in num_cols:
    lenx = len(df.loc[df[col] == 0 ])
    if lenx > 0:
        print(f"{col} : {lenx}" )
## 11 samples has 0 values in the tenure and toalcharges variables. Delete them. Becaıuse our dataset has enogh samples.
df = df[~(df["tenure"] == 0)]

# delete customerid
df.drop("customerid", axis=1, inplace=True)
# convert churn values to 0,1
df.loc[df["churn"] == 'Yes',"churn"] = 1
df.loc[df["churn"] == "No","churn"] = 0
df["churn"] = df["churn"].astype(int)
df.head()

# Stage 2 : Create new variables

# Stage 3 : run encoding for categorical variables.
ohe_cols = [col for col in cat_cols if 'churn' not in col]
for col in ohe_cols:
    df = one_hot_encoder(df,ohe_cols, drop_first=True)

df.columns

# Stage 4 : Make Standardization for num_cols
rs = RobustScaler()
for col in num_cols:
    df[col] = rs.fit_transform(df[[col]])


df.head()
df["churn"].value_counts(normalize=True)

################################################################
# DUTY 2 : Crete Models
#################################################################
## for all model metric values
model_metrics = pd.DataFrame(data=[])
def add_model_metric(xmodel, xauc, xaccuracy , xf1 ):
    new_row = {"Model" : xmodel, "Auc" : xauc , "Accuracy" : xaccuracy, "F1" : xf1}
    return model_metrics.append(new_row,ignore_index=True)

# prepare data
y = df["churn"]
X = df.drop(["churn"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17, stratify=y)

########################################################################
## fit randomForest Holdout
########################################################################
    rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]

    print("Model : Random Forest Classifer ")
    print("auc score : " + str(roc_auc_score(y_test, y_prob)))
    print("accuracy score : " + str(accuracy_score(y_pred, y_test)))
    print("f1 score : " + str(f1_score(y_pred = y_pred, y_true = y_test)))

    model_metrics = add_model_metric("Random Forest Classifer with Holdout",
                     roc_auc_score(y_test, y_prob),
                     accuracy_score(y_pred, y_test),
                     f1_score(y_pred = y_pred, y_true = y_test))

#######################################################################
## fit randomForest 5-Fold CrossValidation
########################################################################
rf_model = RandomForestClassifier(random_state=17)
cv_results = cross_validate(rf_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

print("Model : Random Forest Classifer ")
print("Validation Method : 10 Fold Cross Validation")
print("auc score : " + str(cv_results['test_roc_auc'].mean()))
print("accuracy score : " + str(cv_results['test_accuracy'].mean()))
print("f1 score : " + str(cv_results['test_f1'].mean()))

model_metrics = add_model_metric("Random Forest Classifer",
                 cv_results['test_roc_auc'].mean(),
                 cv_results['test_accuracy'].mean(),
                 cv_results['test_f1'].mean())
########################################################################
## fit cart model 10-fold CrossValidation
########################################################################
cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)
cv_results = cross_validate(cart_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

print("Model : CART Classifer ")
print("Validation Method : 10 Fold Cross Validation")
print("auc score : " + str(cv_results['test_roc_auc'].mean()))
print("accuracy score : " + str(cv_results['test_accuracy'].mean()))
print("f1 score : " + str(cv_results['test_f1'].mean()))

model_metrics = add_model_metric("CART Classifer",
                 cv_results['test_roc_auc'].mean(),
                 cv_results['test_accuracy'].mean(),
                 cv_results['test_f1'].mean())
########################################################################
## fit GBM model 10-fold CrossValidation
########################################################################
gbm_model = GradientBoostingClassifier(random_state=17)
cv_results = cross_validate(gbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

print("Model : GBM Classifer ")
print("Validation Method : 10 Fold Cross Validation")
print("auc score : " + str(cv_results['test_roc_auc'].mean()))
print("accuracy score : " + str(cv_results['test_accuracy'].mean()))
print("f1 score : " + str(cv_results['test_f1'].mean()))

model_metrics = add_model_metric("GBM Classifer",
                 cv_results['test_roc_auc'].mean(),
                 cv_results['test_accuracy'].mean(),
                 cv_results['test_f1'].mean())
########################################################################
## fit XGBoost model 10-fold CrossValidation
########################################################################
xgboost_model = XGBClassifier(random_state=17, use_label_encoder=False)
cv_results = cross_validate(xgboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

print("Model : XGBoost Classifer ")
print("Validation Method : 10 Fold Cross Validation")
print("auc score : " + str(cv_results['test_roc_auc'].mean()))
print("accuracy score : " + str(cv_results['test_accuracy'].mean()))
print("f1 score : " + str(cv_results['test_f1'].mean()))

model_metrics = add_model_metric("XGBoost Classifer",
                 cv_results['test_roc_auc'].mean(),
                 cv_results['test_accuracy'].mean(),
                 cv_results['test_f1'].mean())

########################################################################
## fit LightGBM  model 10-fold CrossValidation
########################################################################
df.columns = df.columns.str.replace(" ","")
lgbm_model = LGBMClassifier(random_state=17)
cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

print("Model : LightGBM Classifer ")
print("Validation Method : 10 Fold Cross Validation")
print("auc score : " + str(cv_results['test_roc_auc'].mean()))
print("accuracy score : " + str(cv_results['test_accuracy'].mean()))
print("f1 score : " + str(cv_results['test_f1'].mean()))

model_metrics = add_model_metric("LightGBM Classifer",
                 cv_results['test_roc_auc'].mean(),
                 cv_results['test_accuracy'].mean(),
                 cv_results['test_f1'].mean())
########################################################################
## fit CatBoost  model 10-fold CrossValidation
########################################################################
catboost_model = CatBoostClassifier(random_state=17, verbose=False)
cv_results_train = cross_validate(catboost_model, X_train, y_train, cv=5, scoring=["accuracy", "f1", "roc_auc"])


print("Model : CatBoost Classifer ")
print("Validation Method : 10 Fold Cross Validation")
print("auc score : " + str(cv_results['test_roc_auc'].mean()))
print("accuracy score : " + str(cv_results['test_accuracy'].mean()))
print("f1 score : " + str(cv_results['test_f1'].mean()))

model_metrics = add_model_metric("CatBoost Classifer",
                 cv_results['test_roc_auc'].mean(),
                 cv_results['test_accuracy'].mean(),
                 cv_results['test_f1'].mean())


print(model_metrics.sort_values("Accuracy",ascending=False))
###################################################
## Choose four model according to the accuracy score.
## Models
# 1. GBM 2. CatBoost 3. LightGBM 4.Random Forest
###################################################
gbm_model2 = GradientBoostingClassifier(random_state=17)
gbm_model2.get_params()
""" bu kadar paramere için bilgisayarım cevap vermiyor.
gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}
"""
gbm_params = {"max_depth": [3, 5, 1],"n_estimators": [100, 200, 6000]}
gbm_best_grid = GridSearchCV(gbm_model2, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
gbm_best_grid.best_params_
gbm_final = gbm_model2.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)

cv_results = cross_validate(gbm_best_grid, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

model_metrics = add_model_metric("GBM Classifer with HyperParameters",
                 cv_results['test_roc_auc'].mean(),
                 cv_results['test_accuracy'].mean(),
                 cv_results['test_f1'].mean())

print(model_metrics.sort_values('Model'))

###############################################################
# 2. CatBoost HyperParameter Settings
###############################################################
catboost_model = CatBoostClassifier(random_state=17, verbose=False)
catboost_model.get_params()
gbm_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

gbm_best_grid = GridSearchCV(catboost_model, gbm_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)
gbm_best_grid.best_params_
gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)

cv_results = cross_validate(gbm_best_grid, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

model_metrics = add_model_metric("CatBoost Classifer with HyperParameters",
                 cv_results['test_roc_auc'].mean(),
                 cv_results['test_accuracy'].mean(),
                 cv_results['test_f1'].mean())

model_metrics.sort_values('Model')

###############################################################
# 3. LightGBM HyperParameter Settings
###############################################################
lgbm_model = LGBMClassifier(random_state=17, colsample_bytree=0.9, learning_rate=0.01)
lgbm_model.get_params()
lgbm_params = {"n_estimators": [200, 400, 1000, 5000, 8000, 9000, 10000]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)
cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

model_metrics = add_model_metric("LightGBM Classifer with HyperParameters",
                 cv_results['test_roc_auc'].mean(),
                 cv_results['test_accuracy'].mean(),
                 cv_results['test_f1'].mean())

model_metrics.sort_values('Model')

###############################################################
# 4. Random Forest HyperParameter Settings
###############################################################
rf_model = RandomForestClassifier(random_state=17)
rf_model.get_params()
rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

model_metrics = add_model_metric("Random Forest Classifer with HyperParameters",
                 cv_results['test_roc_auc'].mean(),
                 cv_results['test_accuracy'].mean(),
                 cv_results['test_f1'].mean())

model_metrics.sort_values('Model')

########################################################################
# feature importance
X_importance = X_test

# Explain model predictions using shap library:
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_importance)
shap.summary_plot(shap_values, X_importance, plot_type='bar')
shap.dependence_plot(ind="tenure", shap_values, features=X_importance)