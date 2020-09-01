# python_machine-learning
## 分類
### 羅吉斯迴歸from sklearn.linear_model import LogisticRegression
#### 參數
penalty:"l1","l2","elasticnet","none"    
dual:bool,default=False (l2 penalty with liblinear solver)  
tol:float,default=1e-4  
C:float, default=1.0  
fit_intercept:bool, default=True  
intercept_scaling:float, default=1 (liblinear)  
class_weight:dict or ‘balanced’, default=None  
solve:"newton-cg","lbfgs","liblinear","sag","saga", default="lbfgs"  
max_iter:int, default=100  
multi_class:"auto", "ovr", "multinomial", default="auto"  
verbose:int, default=0  
warm_start:bool, default=False  
n_jobs:int, default=None  
l1_ratio:float, default=None  
#### 方法
decision_function(X)  
fit(X, y\[, sample_weight\])  
get_params(\[deep\])  
predict(X)  
predict_log_proba(X)  
predict_proba(X)  
score(X, y\[, sample_weight\])  
set_params(\*\*params)  
densify()  
sparsify()  
#### 屬性
classes_  
coef_  
intercept_  
n_iter_  
### 感知器
#### 參數
penalty:"l1","l2","elasticnet","none"   
alpha:float, default=0.0001  
fit_intercept:bool, default=True  
max_iter:int, default=100  
tol:float, default=1e-3  
verbose:int, default=0  
shuffle:bool, default=True  
warm_start:bool, default=False  
eta0:double, default=1  
n_jobs:int, default=None  
random_state:int, RandomState instance, default=None  
early_stopping:bool, default=False  
validation_fraction:float, default=0.1  
n_iter_no_change:int, default=5  
class_weight:dict, {class_label: weight} or “balanced”, default=None  
warm_start:bool, default=False  
#### 方法
decision_function(X)  
fit(X, y\[, coef_init, intercept_init, …\])  
get_params(\[deep\])  
predict(X)  
partial_fit(X, y\[, classes, sample_weight\])  
score(X, y\[, sample_weight\])  
set_params(\*\*params)  
densify()  
sparsify()  
#### 屬性
classes_  
coef_  
intercept_  
n_iter_  
t_  
### 隨機梯度下降法from sklearn.linear_model import SGDClassifier
*SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None)=Perceptron()  
#### 參數
loss:str, default=’hinge’  
penalty:"l1","l2","elasticnet","none"    
alpha:float, default=0.0001  
l1_ratio:float, default=0.15  
fit_intercept:bool, default=True  
max_iter:int, default=100  
tol:float, default=1e-3  
verbose:int, default=0  
epsilon:float, default=0.1  
n_jobs:int, default=None   
random_state:int, RandomState instance, default=None   
early_stopping:bool, default=False  
learning_rate:str, default=’optimal’  
eta0:double, default=0.0  
power_t:double, default=0.5  
validation_fraction:float, default=0.1  
n_iter_no_change:int, default=5  
class_weight:dict, {class_label: weight} or “balanced”, default=None  
warm_start:bool, default=False  
average:bool or int, default=False  
#### 方法  
decision_function(X)  
fit(X, y\[, coef_init, intercept_init, …\])  
get_params(\[deep\])  
predict(X)  
partial_fit(X, y\[, classes, sample_weight\])  
score(X, y\[, sample_weight\])  
set_params(\*\*params)  
densify()  
sparsify()  
#### 屬性
classes_  
coef_  
intercept_  
n_iter_  
t_  
loss_function_  
### KNN最近鄰模型 from sklearn.neighbors import KNeighborsClassifier
#### 參數
n_neighbors:int, default=5  
weights:"uniform", "distance" or callable, default="uniform"  
algorithm:"auto", "ball_tree", "kd_tree", "brute", default="auto"  
leaf_size:int, default=30 ("ball_tree", "kd_tree")  
p:int, default=2 (Power parameter for the Minkowski metric)  
metric:str or callable, default="minkowski"  
metric_params:dict, default=None  
n_jobs:int, default=None  
#### 方法
fit(X, y)  
get_params(\[deep\])  
predict(X)  
predict_proba(X)  
score(X, y\[, sample_weight\])  
set_params(\*\*params)  
kneighbors(\[X, n_neighbors, return_distance\])  
kneighbors_graph(\[X, n_neighbors, mode\])  
#### 屬性
classes_  
effective_metric_  
effective_metric_params_  
outputs_2d_  
### 樸素貝氏from sklearn.naive_bayes import MultinomialNB,BernoulliNB,ComplementNB
#### 參數
alpha:float, default=1.0  
fit_prior:bool, default=True  
class_prior:array-like of shape (n_classes,), default=None  
binarize:float or None, default=0.0(BernoulliNB)  
norm:bool, default=False(ComplementNB)  
#### 方法
fit(X, y\[, sample_weight\])  
get_params(\[deep\])  
partial_fit(X, y\[, classes, sample_weight\])  
predict(X)  
predict_log_proba(X)  
predict_proba(X)  
score(X, y\[, sample_weight\])  
set_params(\*\*params)  
#### 屬性
class_count_  
class_log_prior_  
classes_  
coef_(MultinomialNB)  
feature_all_(ComplementNB)  
feature_count_  
feature_log_prob_  
intercept_(MultinomialNB)  
n_features_  
### 高斯貝氏from sklearn.naive_bayes import GaussianNB
#### 參數
priors:array-like of shape (n_classes,)  
var_smoothing:float, default=1e-9  
#### 方法
fit(X, y\[, sample_weight\])  
get_params(\[deep\])  
partial_fit(X, y\[, classes, sample_weight\])  
predict(X)  
predict_log_proba(X)  
predict_proba(X)  
score(X, y\[, sample_weight\])  
set_params(\*\*params)  
#### 屬性
class_count_  
class_prior_  
classes_  
epsilon_  
sigma_  
theta_  
### 支持向量機from sklearn.svm import SVC,NuSVC
#### 參數
C:float, default=1.0(SVC),nu:float, default=0.5(NuSVC)  
kernel:"linear", "poly", "rbf", "sigmoid", "precomputed", default="rbf"  
degree:int, default=3("poly")  
gamma:"scale", "auto" or float, default="scale" (‘rbf’, ‘poly’ and ‘sigmoid’)  
coef0:float, default=0.0("poly" and "sigmoid")  
shrinking:bool, default=True  
probability:bool, default=False  
tol:float, default=1e-3  
cache_size:float, default=200  
class_weight:dict or "balanced", default=None  
verbose:bool, default=False  
max_iter:int, default=-1  
decision_function_shape:"ovo", "ovr", default="ovr"  
break_ties:bool, default=False  
random_state:int or RandomState instance, default=None  
#### 方法
decision_function(X)  
fit(X, y\[, sample_weight\])  
get_params(\[deep\])  
predict(X)  
score(X, y\[, sample_weight\])  
set_params(\*\*params)  
#### 屬性
support_  
support_vectors_  
n_support_  
dual_coef_  
coef_  
intercept_  
fit_status_  
classes_  
probA_  
probB_  
class_weight_  
shape_fit_  
### 支持向量機from sklearn.svm import LinearSVC
#### 參數
penalty:"l1", "l2", default="l2"  
loss:"hinge", "squared_hinge", default="squared_hinge"  
dual:bool, default=True  
tol:float, default=1e-4  
C:float, default=1.0  
multi_class:"ovr", "crammer_singer", default="ovr"  
fit_intercept:bool, default=True  
intercept_scaling:float, default=1  
class_weight:dict or "balanced", default=None  
verbose:int, default=0  
random_state:int or RandomState instance, default=None  
max_iter:int, default=1000  
#### 方法
decision_function(X)   
fit(X, y\[, sample_weight\])  
get_params(\[deep\])  
predict(X)   
score(X, y\[, sample_weight\])  
set_params(\*\*params)  
densify()  
sparsify()  
#### 屬性
coef_  
intercept_  
classes_  
n_iter_  
### 決策樹from sklearn.tree import DecisionTreeClassifier
#### 參數
criterion:"gini", "entropy", default="gini"  
splitter:"best", "random", default="best"  
max_depth:int, default=None  
min_samples_split:int or float, default=2  
min_samples_leaf:int or float, default=1  
min_weight_fraction_leaf:float, default=0.0  
max_features:int, float or {“auto”, “sqrt”, “log2”}, default=None  
random_state:int, RandomState instance, default=None  
max_leaf_nodes:int, default=None  
min_impurity_decrease:float, default=0.0  
min_impurity_split:float, default=0  
class_weight:dict, list of dict or "balanced", default=None  
ccp_alpha:non-negative float, default=0.0  
#### 方法
apply(X\[, check_input\])  
cost_complexity_pruning_path(X, y\[, …\])  
decision_path(X\[, check_input\])  
fit(X, y\[, sample_weight, check_input, …\])  
get_depth()  
get_n_leaves()  
get_params(\[deep\])  
predict(X\[, check_input\])  
predict_log_proba(X)  
predict_proba(X\[, check_input\])  
score(X, y\[, sample_weight\])    
set_params(\*\*params)  
#### 屬性  
classes_  
feature_importances_  
max_features_  
n_classes_   
n_features_  
n_outputs_  
tree_  
### 隨機森林from sklearn.ensemble import RandomForestClassifier
#### 參數
n_estimators:int, default=100  
criterion:"gini", "entropy", default="gini"  
max_depth:int, default=None  
min_samples_split:int or float, default=2  
min_samples_leaf:int or float, default=1  
min_weight_fraction_leaf:float, default=0.0  
max_features:int, float or {“auto”, “sqrt”, “log2”}, default=None  
random_state:int, RandomState instance, default=None  
max_leaf_nodes:int, default=None  
min_impurity_decrease:float, default=0.0  
min_impurity_split:float, default=0   
bootstrap:bool, default=True  
oob_score:bool, default=False  
n_jobs:int, default=None  
random_state:int or RandomState, default=None  
verbose:int, default=0  
warm_start:bool, default=False   
class_weight:"balanced", "balanced_subsample”, dict or list of dicts, default=None  
ccp_alpha:non-negative float, default=0.0  
max_samples:int or float, default=None  
#### 方法
apply(X\[, check_input\])   
decision_path(X\[, check_input\])   
fit(X, y\[, sample_weight, check_input, …\])  
get_params(\[deep\])  
predict(X\[, check_input\])  
predict_log_proba(X)  
predict_proba(X\[, check_input\])  
score(X, y\[, sample_weight\])  
set_params(\*\*params)  
#### 屬性
base_estimator_  
estimators_  
classes_   
n_classes_  
n_features_  
n_outputs_  
oob_score_  
oob_decision_function_  
### bagging
#### 參數
base_estimator:object, default=None  
n_estimators:int, default=10  
max_samples:int or float, default=1.0  
max_features:int or float, default=1.0  
bootstrap:bool, default=True  
bootstrap_features:bool, default=False  
oob_score:bool, default=False  
warm_start:bool, default=False  
n_jobs:int, default=None   
random_state:int or RandomState, default=None  
verbose:int, default=0  
#### 方法
decision_path(X\[, check_input\])  
fit(X, y\[, sample_weight, check_input, …\])  
get_params(\[deep\])  
predict(X\[, check_input\])  
predict_log_proba(X)  
predict_proba(X\[, check_input\])  
score(X, y\[, sample_weight\])  
set_params(\*\*params)  
#### 屬性
base_estimator_  
n_features_  
estimators_  
estimators_samples_  
estimators_features_  
classes_  
n_classes_  
oob_score_  
oob_decision_function_  
## adaboosting
#### 參數
base_estimator:object, default=None  
n_estimators:int, default=50  
learning_rate:float, default=1  
algorithm:"SAMME", "SAMME.R", default="SAMME.R"  
random_state:int or RandomState, default=None  
#### 方法
decision_path(X\[, check_input\])  
fit(X, y\[, sample_weight, check_input, …\])  
get_params(\[deep\])  
predict(X\[, check_input\])  
predict_log_proba(X)  
predict_proba(X\[, check_input\])  
score(X, y\[, sample_weight\])  
set_params(\*\*params)  
staged_decision_function(X)  
staged_predict(X)  
staged_predict_proba(X)  
staged_score(X, y\[, sample_weight\])  
#### 屬性
base_estimator_  
estimators_   
classes_  
n_classes_  
estimator_weights_  
estimator_errors_   
feature_importances_  
## gradientboosting
#### 參數
loss:"deviance", "exponential", default="deviance"  
learning_rate:float, default=0.1  
n_estimators:int, default=100   
subsample:float, default=1.0  
criterion:"friedman_mse", "mse", "mae", default="friedman_mse"  
min_samples_split:int or float, default=2  
min_samples_leaf:int or float, default=1  
min_weight_fraction_leaf:float, default=0.0  
max_depth:int, default=3  
min_impurity_decrease:float, default=0.0  
min_impurity_split:float, default=None  
init:estimator or ‘zero’, default=None
random_state:int or RandomState, default=None  
max_features:"auto", "sqrt", "log2", int or float, default=None  
verbose:int, default=0  
max_leaf_nodes:int, default=None  
warm_start:bool, default=False  
validation_fraction:float, default=0.1  
n_iter_no_change:int, default=None  
tol:float, default=1e-4  
ccp_alpha:non-negative float, default=0.0  
#### 方法  

apply(X)  
decision_function(X)   
fit(X, y\[, sample_weight, monitor\])  
get_params(\[deep\])  
predict(X\[, check_input\])  
predict_log_proba(X)  
predict_proba(X\[, check_input\])  
score(X, y\[, sample_weight\])  
set_params(\*\*params)  
staged_decision_function(X)    
staged_predict(X)  
staged_predict_proba(X)  
#### 屬性
n_estimators_  
feature_importances_  
oob_improvement_  
train_score_  
loss_  
init_  
estimators_  
classes_  
n_features_  
n_classes_  
max_features_  
### 其他模型
from xgboost import XGBClassifier  
from catboost import CatBoostClassifier  
import lightgbm
from mlxtend.classifier import StackingClassifier   
## 預測數值
### 線性模型 from sklearn.linear_model import LinearRegression
#### 參數
fit_intercept:bool, default=True   
normalize:bool, default=False  
copy_X:bool, default=True  
n_jobs:int, default=None  
#### 方法
fit(X, y\[, sample_weight\])  
get_params(\[deep\])  
predict(X)  
score(X, y\[, sample_weight\])  
set_params(\*\*params)  
#### 屬性
coef_  
rank_  
singular_  
intercept_  
  
### 脊迴歸 from sklearn.linear_model import Ridge
#### 參數
alpha:float, ndarray of shape (n_targets,), default=1.0  
fit_intercept:bool, default=True   
normalize:bool, default=False  
copy_X:bool, default=True  
max_iter:int, default=None  
tol:float, default=1e-3  
solver:"auto", "svd", "cholesky", ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’, default=’auto’  
random_state:int, RandomState instance, default=None  
#### 方法 
fit(X, y\[, sample_weight\])  
get_params(\[deep\])     
predict(X)    
score(X, y\[, sample_weight\])  
set_params(\*\*params)  
#### 屬性
coef_  
intercept_  
n_iter_  
### LASSO from sklearn.linear_model import Lasso
#### 參數  
alpha:float, ndarray of shape (n_targets,), default=1.0  
fit_intercept:bool, default=True  
normalize:bool, default=False  
precompute:‘auto’, bool or array-like of shape (n_features, n_features), default=False  
copy_X:bool, default=True   
max_iter:int, default=None  
tol:float, default=1e-3  
warm_start:bool, default=False  
positive:bool, default=False  
random_state:int, RandomState instance, default=None   
selection:‘cyclic’, ‘random’, default=’cyclic’  
#### 方法
fit(X, y\[, sample_weight\])  
get_params(\[deep\])  
path(X, y, *\[, l1_ratio, eps, n_alphas, …\])  
predict(X)    
score(X, y\[, sample_weight\])    
set_params(\*\*params)   
#### 屬性    
coef_  
intercept_  
n_iter_  
sparse_coef_  
### ElasticNet from sklearn.linear_model import ElasticNet
#### 參數
alpha:float, ndarray of shape (n_targets,), default=1.0  
l1_ratio:float, default=0.5  
fit_intercept:bool, default=True  
normalize:bool, default=False  
precompute:‘auto’, bool or array-like of shape (n_features, n_features), default=False  
copy_X:bool, default=True  
max_iter:int, default=None   
tol:float, default=1e-3  
warm_start:bool, default=False  
positive:bool, default=False  
random_state:int, RandomState instance, default=None   
selection:‘cyclic’, ‘random’, default=’cyclic’   
#### 方法
fit(X, y\[, sample_weight\])  
get_params(\[deep\])  
path(X, y, *\[, l1_ratio, eps, n_alphas, …\])  
predict(X)  
score(X, y\[, sample_weight\])    
set_params(\*\*params)  
#### 屬性
coef_  
intercept_  
n_iter_  
sparse_coef_  
### KNN最近鄰模型 from sklearn.neighbors import KNeighborsRegressor
#### 參數
n_neighbors:int, default=5  
weights:"uniform", "distance" or callable, default="uniform"  
algorithm:"auto", "ball_tree", "kd_tree", "brute", default="auto"  
leaf_size:int, default=30 ("ball_tree", "kd_tree")  
p:int, default=2 (Power parameter for the Minkowski metric)  
metric:str or callable, default="minkowski"  
metric_params:dict, default=None  
n_jobs:int, default=None  
#### 方法
fit(X, y)  
get_params(\[deep\])  
predict(X)  
score(X, y\[, sample_weight\])  
set_params(\*\*params)  
kneighbors(\[X, n_neighbors, return_distance\])  
kneighbors_graph(\[X, n_neighbors, mode\])  
#### 屬性
effective_metric_
effective_metric_params_
### 決策樹迴歸from sklearn.tree import DecisionTreeRegressor
#### 參數
criterion:"mse", "friedman_mse", "mae", default="mse"  
splitter:"best", "random", default="best"  
max_depth:int, default=None  
min_samples_split:int or float, default=2  
min_samples_leaf:int or float, default=1  
min_weight_fraction_leaf:float, default=0.0  
max_features:int, float or {“auto”, “sqrt”, “log2”}, default=None  
random_state:int, RandomState instance, default=None  
max_leaf_nodes:int, default=None  
min_impurity_decrease:float, default=0.0  
min_impurity_split:float, default=0  
ccp_alpha:non-negative float, default=0.0  
#### 方法
apply(X\[, check_input\])  
cost_complexity_pruning_path(X, y\[, …\])  
decision_path(X\[, check_input\])  
fit(X, y\[, sample_weight, check_input, …\])  
get_depth()  
get_n_leaves()  
get_params(\[deep\])  
predict(X\[, check_input\])  
score(X, y\[, sample_weight\])  
set_params(\*\*params)  
#### 屬性
feature_importances_  
max_features_  
n_features_  
n_outputs_  
tree_  
## 隨機森林迴歸from sklearn.ensemble import RandomForestRegressor
#### 參數
n_estimators:int, default=100  
criterion:"mse", "mae", default="mse"  
max_depth:int, default=None  
min_samples_split:int or float, default=2  
min_samples_leaf:int or float, default=1  
min_weight_fraction_leaf:float, default=0.0  
max_features:int, float or {“auto”, “sqrt”, “log2”}, default=None  
random_state:int, RandomState instance, default=None  
max_leaf_nodes:int, default=None  
min_impurity_decrease:float, default=0.0  
min_impurity_split:float, default=0  
bootstrap:bool, default=True  
oob_score:bool, default=False  
n_jobs:int, default=None  
verbose:int, default=0  
warm_start:bool, default=False  
ccp_alpha:non-negative float, default=0.0  
max_samples:int or float, default=None  
#### 方法
apply(X\[, check_input\])  
decision_path(X\[, check_input\])  
fit(X, y\[, sample_weight, check_input, …\])  
get_params(\[deep\])  
predict(X\[, check_input\])   
score(X, y\[, sample_weight\])   
set_params(\*\*params)  
#### 屬性
base_estimator_   
estimators_  
feature_importances_   
n_features_   
n_outputs_  
oob_score_  
oob_prediction_  
## adaboosting
#### 參數
base_estimator:object, default=None  
n_estimators:int, default=50  
learning_rate:float, default=1  
loss:"linear", "square", "exponential", default="linear"  
random_state:int or RandomState, default=None  
#### 方法
fit(X, y\[, sample_weight\])   
get_params(\[deep\])  
predict(X)  
score(X, y\[, sample_weight\])    
set_params(\*\*params)    
staged_decision_function(X)  
staged_predict(X)  
staged_score(X, y\[, sample_weight\])  
#### 屬性
base_estimator_  
estimators_  
estimator_weights_  
estimator_errors_  
feature_importances_  
## gradientboosting
#### 參數 
loss:"ls","lad", "huber", "quantile", default="ls"  
learning_rate:float, default=0.1  
n_estimators:int, default=100  
subsample:float, default=1.0  
criterion:"friedman_mse", "mse", "mae", default="friedman_mse"   
min_samples_split:int or float, default=2  
min_samples_leaf:int or float, default=1  
min_weight_fraction_leaf:float, default=0.0  
max_depth:int, default=3  
min_impurity_decrease:float, default=0.0  
min_impurity_split:float, default=None  
init:estimator or ‘zero’, default=None  
random_state:int or RandomState, default=None  
max_features:"auto", "sqrt", "log2", int or float, default=None  
alpha:float, default=0.9  
verbose:int, default=0  
max_leaf_nodes:int, default=None  
warm_start:bool, default=False  
validation_fraction:float, default=0.1  
n_iter_no_change:int, default=None  
tol:float, default=1e-4  
ccp_alpha:non-negative float, default=0.0  
#### 方法

apply(X)  
decision_function(X)  
fit(X, y\[, sample_weight, monitor\])  
get_params(\[deep\])  
predict(X\[, check_input\])  
predict_log_proba(X)   
predict_proba(X\[, check_input\])  
score(X, y\[, sample_weight\])  
set_params(\*\*params)  
staged_decision_function(X)  
staged_predict(X)  
staged_predict_proba(X)  
#### 屬性
feature_importances_  
oob_improvement_   
train_score_   
loss_  
init_  
estimators_  
n_features_  
max_features_  
### 其他模型
from mlxtend.regressor import StackingRegressor
## 分群
### 階層樹from sklearn.cluster import AgglomerativeClustering
#### 參數
n_clusters:int or None, default=2  
affinity:str or callable, default=’euclidean’  
memorystr or object with the joblib.Memory interface, default=None  
connectivityarray-like or callable, default=None  
compute_full_tree‘auto’ or bool, default=’auto’  
linkage{“ward”, “complete”, “average”, “single”}, default=”ward”  
distance_thresholdfloat, default=None  
#### 方法
fit(X\[, y\])   
fit_predict(X\[, y\])  
get_params(\[deep\])  
set_params(\*\*params)  
#### 屬性
n_clusters_  
labels_  
n_leaves_  
n_connected_components_  
children_   
### KMEANS from sklearn.cluster import Kmeans
#### 參數
n_clusters:int or None, default=8  
init{‘k-means++’, ‘random’, ndarray, callable}, default=’k-means++’  
n_initint, default=10  
max_iterint, default=300  
tolfloat, default=1e-4  
verboseint, default=0  
random_stateint, RandomState instance, default=None  
copy_xbool, default=True  
n_jobsint, default=None  
algorithm{“auto”, “full”, “elkan”}, default=”auto”  
#### 方法
fit(X\[, y\])  
fit_predict(X\[, y, sample_weight\])  
fit_transform(X\[, y, sample_weight\])   
get_params(\[deep\])  
predict(X\[, sample_weight\])  
score(X\[, y, sample_weight\])  
set_params(\*\*params)  
transform(X)  
#### 屬性
cluster_centers_  
labels_  
inertia_  
n_iter_  
### minibatch KMEANS from sklearn.cluster import MiniBatchKMeans
#### 參數
n_clusters:int or None, default=8  
init{‘k-means++’, ‘random’} or ndarray of shape (n_clusters, n_features), default=’k-means++’  
max_iter:int, default=300  
tol:float, default=1e-4  
batch_size:int, default=100  
verbose:int, default=0  
compute_labels:bool, default=True  
max_no_improvement:int, default=10  
random_state:int, RandomState instance, default=None  
init_size:int, default=None  
n_init:int, default=3  
reassignment_ratio:float, default=0.01  
#### 方法
fit(X\[, y\])  
fit_predict(X\[, y, sample_weight\])  
fit_transform(X\[, y, sample_weight\])  
get_params(\[deep\])  
partial_fit(X\[, y, sample_weight\])   
predict(X\[, sample_weight\])  
score(X\[, y, sample_weight\])  
set_params(\*\*params)   
transform(X)   
#### 屬性
cluster_centers_  
labels_  
inertia_  
### 光譜集群from sklearn.cluster import SpectralClustering
#### 參數
n_clusters:integer, optional  
eigen_solver:{None, ‘arpack’, ‘lobpcg’, or ‘amg’}  
n_components:integer, optional, default=n_clusters  
random_state:int, RandomState instance, default=None  
n_init:int, optional, default: 10  
gamma:float, default=1.0  
affinity:string or callable, default ‘rbf’  
n_neighbors:integer  
eigen_tol:float, optional, default: 0.0  
assign_labels:{‘kmeans’, ‘discretize’}, default: ‘kmeans’  
degree:float, default=3  
coef0:float, default=1  
kernel_params:dictionary of string to any, optional  
n_jobs:int or None, optional (default=None)  
#### 方法
fit(X\[, y\])  
fit_predict(X\[, y\])   
get_params(\[deep\])  
set_params(\*\*params)  
#### 屬性
affinity_matrix_  
labels_  
### DBSCAN from sklearn.cluster import DBSCAN
#### 參數
eps:float, default=0.5  
min_samples:int, default=5  
metric:string, or callable, default=’euclidean’  
metric_paramsdict, default=None  
algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’  
leaf_sizeint, default=30  
pfloat, default=None  
n_jobsint, default=None  
#### 方法
fit(X\[, y, sample_weight\])  
fit_predict(X\[, y, sample_weight\])  
get_params(\[deep\])  
set_params(\*\*params)  
#### 屬性
core_sample_indices_  
components_   
labels_   
## 降維 
### from sklearn.decomposition import PCA
### from sklearn.decomposition import KernelPCA
### from sklearn.decomposition import SparsePCA
### from sklearn.decomposition import MiniBatchSparsePCA
### from sklearn.decomposition import IncrementalPCA
### from sklearn.decomposition import TruncatedSVD
### from sklearn.manifold import MDS
### from sklearn.manifold import LocallyLinearEmbedding
### from sklearn.manifold import TSNE
### from sklearn.manifold import Isomap
### from sklearn.lda import LDA
### from sklearn.decomposition import LatentDirichletAllocation
