import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def check_df(xdf, xrow_count=5, xplot=False):
    print("*************** DATASET INFO ************************")
    print("*************** SHAPE ************************")
    print(xdf.shape)
    print("*************** INFO ************************")
    print(xdf.info())
    print("*************** TIPLER ************************")
    print(xdf.dtypes)
    print("*************** HEAD ************************")
    print(xdf.head(xrow_count))
    print("*************** TAIL ************************")
    print(xdf.tail(xrow_count))
    print("*************** Nan Numbers ************************")
    print(xdf.isnull().sum())
    print("*************** Describe Istatics ************************")
    print(xdf.describe().T)
    print("*************** UNIQUE VALUE NUMBERS ************************")
    print(xdf.nunique())
    print("***************  ************************")
    if xplot:
        xdf.hist()
        plt.show(block=True)

def cat_summary(xdf,xcol,xplot = False):
    print(pd.DataFrame({xcol : xdf[xcol].value_counts(),
                        "Ratio" : 100* xdf[xcol].value_counts() / len(xdf)}))
    if xdf[xcol].dtype == "bool" :
        xdf[xcol] = xdf[xcol].astype(int)
    if xplot :
        sns.countplot(x=xdf[xcol],data=xdf)
        plt.show(block=True)
    print("######################################################")

def num_summary(xdf,xcol,plot = False):
    print(f" Column : {xcol}" )
    quantiles = [0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90]
    print(xdf[xcol].describe(quantiles).T)
    print("################################################")
    if plot:
        xdf[xcol].hist()
        plt.xlabel(xcol)
        plt.title(xcol)
        plt.show(block=True)

def target_summary_with_cat(xdf, xtarget,xcat_col):
    print(pd.DataFrame({"TARGET FREQUENCY ": xdf.groupby(xcat_col)[xtarget].count(),
                        "RATIO ": 100 * xdf.groupby(xcat_col)[xtarget].count() / len(xdf)}))

def target_summary_with_num(xdf, xtarget,xnum_col):
    ## SORU print(xdf.groupby(xtarget)[xnum_col].agg({xnum_col : "mean"}))
    ## NUMERICAL COL DEĞERİNİ YAZDIRAMADIM
    print(xdf.groupby(xtarget)[xnum_col].agg("mean"))

def grab_col_names(xdf,xcat_th = 10 , xcar_th = 20):
    cat_cols = [col for col in xdf.columns if xdf[col].dtypes in ["bool", "object", "category"]]
    num_but_cat = [col for col in xdf.columns if xdf[col].dtypes in ["int64", "float64"]
                   and xdf[col].nunique() < xcat_th]
    cat_but_car = [col for col in xdf.columns if xdf[col].dtypes in ["object", "category"]
                   and xdf[col].nunique() > xcar_th]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in xdf.columns if str(xdf[col].dtypes) in ["float64","int64"]
                and col not in cat_cols]

    print(f"Observations : {xdf.shape[0]}")
    print(f"Variables : {xdf.shape[1]}")
    print(f"Cat_cols : {len(cat_cols)}")
    print(f"Num_cols : {len(num_cols)}")
    print(f"Cat_But_car : {len(cat_but_car)}")
    print(f"Num_but-cat : {len(num_but_cat)}")
    print(f"Categoric Variables : {cat_cols}")
    print(f"Numeric Variables :  {num_cols}")
    
    return cat_cols, num_cols, cat_but_car


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def check_MissingValue(dataframe, col_name):
    if (dataframe[col_name].isnull().sum() > 0):
        return True
    else:
        return False

def outlier_analyser(xdf, xcol_list) :
    print(f"##################### Outlier Analyse #######################")
    for col in xcol_list:
        print(f" {col} : {check_outlier(xdf, col)}")

def missing_value_analyser(xdf, xcol_list):
    print(f"##################### Missing Value Analyse #######################")
    for col in xcol_list:
        print(f" {col} : {check_MissingValue(xdf, col)}")

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def plot_importance(model, features, num=10, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')
    pd.set_option('display.width', 500)