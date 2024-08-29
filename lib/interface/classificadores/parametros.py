from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import streamlit as st


def parametros_arvore_decisao(st, X_train=None, y_train=None):
    # Definir as chaves de estado da sessão para cada hiperparâmetro
    if 'ccp_alpha_b' not in st.session_state:
        st.session_state.ccp_alpha_b = 0.02
    if 'max_depth_tree_b' not in st.session_state:
        st.session_state.max_depth_tree_b = 10
    if 'min_samples_split_tree_b' not in st.session_state:
        st.session_state.min_samples_split_tree_b = 5
    if 'min_samples_leaf_tree_b' not in st.session_state:
        st.session_state.min_samples_leaf_tree_b = 2
    if 'max_features_tree_b' not in st.session_state:
        st.session_state.max_features_tree_b = None
    if 'criterion_tree_b' not in st.session_state:
        st.session_state.criterion_tree_b = "gini"

    with st.sidebar.expander("Hiperparâmetros - Árvores de Decisão", expanded=False):
        if st.button(":rocket: RandomizedSearchCV (Demorado)", key="grid_tree"):
            with st.spinner("..."):
                best = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), param_distributions={
                    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'max_features': ['sqrt', 'log2', None],
                    'criterion': ['gini', 'entropy'],
                    'ccp_alpha': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
                }, n_iter=10, cv=5, n_jobs=-1, verbose=2, refit=True).fit(X_train, y_train)

                st.session_state.ccp_alpha_b = best.best_params_['ccp_alpha']
                st.session_state.max_depth_tree_b = best.best_params_['max_depth']
                st.session_state.min_samples_split_tree_b = best.best_params_['min_samples_split']
                st.session_state.min_samples_leaf_tree_b = best.best_params_['min_samples_leaf']
                st.session_state.max_features_tree_b = best.best_params_['max_features']
                st.session_state.criterion_tree_b = best.best_params_['criterion']

        # Agora use os valores no session_state para os inputs
        ccp_alpha = st.number_input("Complexidade de Podas de Custos", 0.0, 0.5, st.session_state.ccp_alpha_b, key="ccp_alpha_tree")
        max_depth_tree = st.slider("Profundidade Máxima", 1, 20, st.session_state.max_depth_tree_b, key="max_depth_tree")
        min_samples_split_tree = st.slider("Mínimo de Amostras para Split", 2, 20, st.session_state.min_samples_split_tree_b, key="min_samples_split_tree")
        min_samples_leaf_tree = st.slider("Mínimo de Amostras na Folha", 1, 10, st.session_state.min_samples_leaf_tree_b, key="min_samples_leaf_tree")
        max_features_tree = st.selectbox("Máximo de Features", ['sqrt', 'log2', None], index=['sqrt', 'log2', None].index(st.session_state.max_features_tree_b), key="max_features_tree")
        criterion_tree = st.selectbox("Critério", ["gini", "entropy", "log_loss"], index=["gini", "entropy", "log_loss"].index(st.session_state.criterion_tree_b), key="criterion_tree")

    return max_depth_tree, min_samples_split_tree, min_samples_leaf_tree, max_features_tree, criterion_tree, ccp_alpha


def parametros_svm(st, X_train=None, y_train=None):
    # Definir as chaves de estado da sessão para cada hiperparâmetro
    if 'C_svm_g' not in st.session_state:
        st.session_state.C_svm_g = 0.08
    if 'kernel_svm_g' not in st.session_state:
        st.session_state.kernel_svm_g = 'sigmoid'
    if 'degree_svm_g' not in st.session_state:
        st.session_state.degree_svm_g = 3
    if 'gamma_svm_g' not in st.session_state:
        st.session_state.gamma_svm_g = 'scale'
    if 'coef0_svm_g' not in st.session_state:
        st.session_state.coef0_svm_g = 0.50

    with st.sidebar.expander("Hiperparâmetros - SVM", expanded=False):
        if st.button(":rocket: RandomizedSearchCV (Demorado)", key="grid_svm"):
            with st.spinner("..."):
                best = RandomizedSearchCV(SVC(random_state=42), param_distributions={
                    'C': [0.01, 0.1, 1.0, 10.0],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'degree': [2, 3, 4, 5],
                    'gamma': ['scale', 'auto'],
                    'coef0': [0.0, 0.25, 0.50, 0.75, 1.0]
                }, n_iter=10, cv=5, n_jobs=-1, verbose=2, refit=True).fit(X_train, y_train)
                st.session_state.C_svm_g = best.best_params_['C']
                st.session_state.kernel_svm_g = best.best_params_['kernel']
                st.session_state.degree_svm_g = best.best_params_['degree']
                st.session_state.gamma_svm_g = best.best_params_['gamma']
                st.session_state.coef0_svm_g = best.best_params_['coef0']

        C_svm = st.number_input("C (Regularização)", 0.01, 10.0, st.session_state.C_svm_g, key="C_svm")
        kernel_svm = st.selectbox("Kernel", ['linear', 'poly', 'rbf', 'sigmoid'], key="kernel_svm", index=['linear', 'poly', 'rbf', 'sigmoid'].index(st.session_state.kernel_svm_g))
        degree_svm = st.slider("Grau (se kernel for poly)", 2, 5, st.session_state.degree_svm_g, key="degree_svm")
        gamma_svm = st.selectbox("Gamma", ['scale', 'auto'], key="gamma_svm", index=['scale', 'auto'].index(st.session_state.gamma_svm_g))
        coef0_svm = st.slider("Coef0 (para poly/sigmoid)", 0.0, 1.0, st.session_state.coef0_svm_g, key="coef0_svm")

    return C_svm, kernel_svm, degree_svm, gamma_svm, coef0_svm

def parametros_random_forest(st, X_train=None, y_train=None):
    # Definir as chaves de estado da sessão para cada hiperparâmetro
    if 'n_estimators_rf_b' not in st.session_state:
        st.session_state.n_estimators_rf_b = 100
    if 'max_depth_rf_b' not in st.session_state:
        st.session_state.max_depth_rf_b = 10
    if 'min_samples_split_rf_b' not in st.session_state:
        st.session_state.min_samples_split_rf_b = 2
    if 'min_samples_leaf_rf_b' not in st.session_state:
        st.session_state.min_samples_leaf_rf_b = 1
    if 'max_features_rf_b' not in st.session_state:
        st.session_state.max_features_rf_b = None
    if 'criterion_rf_b' not in st.session_state:
        st.session_state.criterion_rf_b = "gini"
    if 'bootstrap_rf_b' not in st.session_state:
        st.session_state.bootstrap_rf_b = True

    with st.sidebar.expander("Hiperparâmetros - Random Forest", expanded=False):
        if st.button(":rocket: RandomizedSearchCV (Demorado)", key="grid_rf"):
            with st.spinner("..."):
                best = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions={
                    'n_estimators': [10, 50, 100, 200],
                    'max_depth': [1, 5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10, 15, 20],
                    'min_samples_leaf': [1, 2, 3, 4, 5],
                    'max_features': ['sqrt', 'log2', None],
                    'criterion': ['gini', 'entropy'],
                    'bootstrap': [True, False]
                }, n_iter=30, cv=5, n_jobs=-1, verbose=2, refit=True).fit(X_train, y_train)
                st.session_state.n_estimators_rf_b = best.best_params_['n_estimators']
                st.session_state.max_depth_rf_b = best.best_params_['max_depth']
                st.session_state.min_samples_split_rf_b = best.best_params_['min_samples_split']
                st.session_state.min_samples_leaf_rf_b = best.best_params_['min_samples_leaf']
                st.session_state.max_features_rf_b = best.best_params_['max_features']
                st.session_state.criterion_rf_b = best.best_params_['criterion']
                st.session_state.bootstrap_rf_b = best.best_params_['bootstrap']

        n_estimators_rf = st.slider("Número de Árvores (Estimadores)", 10, 200, st.session_state.n_estimators_rf_b, key="n_estimators_rf")
        max_depth_rf = st.slider("Profundidade Máxima", 1, 20, st.session_state.max_depth_rf_b, key="max_depth_rf")
        min_samples_split_rf = st.slider("Mínimo de Amostras para Split", 2, 20, st.session_state.min_samples_split_rf_b, key="min_samples_split_rf")
        min_samples_leaf_rf = st.slider("Mínimo de Amostras na Folha", 1, 10, st.session_state.min_samples_leaf_rf_b, key="min_samples_leaf_rf")
        max_features_rf = st.selectbox("Máximo de Features", ['sqrt', 'log2', None], key="max_features_rf", index=['sqrt', 'log2', None].index(st.session_state.max_features_rf_b))
        criterion_rf = st.selectbox("Critério", ["gini", "entropy","log_loss"], key="criterion_rf", index=["gini", "entropy","log_loss"].index(st.session_state.criterion_rf_b))
        bootstrap_rf = st.selectbox("Bootstrap", [True, False], key="bootstrap_rf", index=[True, False].index(st.session_state.bootstrap_rf_b))

    return n_estimators_rf, max_depth_rf, min_samples_split_rf, min_samples_leaf_rf, max_features_rf, criterion_rf, bootstrap_rf

def parametros_logistic_regression(st, X_train=None, y_train=None):
    # Definir as chaves de estado da sessão para cada hiperparâmetro
    if 'C_lr_g' not in st.session_state:
        st.session_state.C_lr_g = 0.01
    if 'solver_lr_g' not in st.session_state:
        st.session_state.solver_lr_g = 'liblinear'

    with st.sidebar.expander("Hiperparâmetros - Logistic Regression", expanded=False):
        if st.button(":rocket: RandomizedSearchCV (Demorado)", key="grid_lr"):
            with st.spinner("..."):
                best = RandomizedSearchCV(LogisticRegression(random_state=42), param_distributions={
                    'C': [0.01, 0.1, 1.0, 10.0],
                    'solver': ['liblinear', 'lbfgs', 'sag', 'saga', 'newton-cg']
                }, n_iter=10, cv=5, n_jobs=-1, verbose=2, refit=True).fit(X_train, y_train)
                st.session_state.C_lr_g = best.best_params_['C']
                st.session_state.solver_lr_g = best.best_params_['solver']

        C_lr = st.number_input("C (Regularização Inversa)", 0.01, 10.0, st.session_state.C_lr_g, key="C_lr")
        solver_lr = st.selectbox("Solver", ['liblinear', 'lbfgs', 'sag', 'saga', 'newton-cg'], key="solver_lr", index=['liblinear', 'lbfgs', 'sag', 'saga', 'newton-cg'].index(st.session_state.solver_lr_g))

    return C_lr, solver_lr


def parametros_knn(st, X_train=None, y_train=None):
    # Definir as chaves de estado da sessão para cada hiperparâmetro
    if 'leaf_size_knn_b' not in st.session_state:
        st.session_state.leaf_size_knn_b = 30
    if 'n_neighbors_knn_b' not in st.session_state:
        st.session_state.n_neighbors_knn_b = 20
    if 'weights_knn_b' not in st.session_state:
        st.session_state.weights_knn_b = 'distance'
    if 'algorithm_knn_b' not in st.session_state:
        st.session_state.algorithm_knn_b = 'brute'

    with st.sidebar.expander("Hiperparâmetros - KNN", expanded=False):
        if st.button(":rocket: RandomizedSearchCV (Demorado)", key="grid_knn"):
            with st.spinner("..."):
                best = RandomizedSearchCV(KNeighborsClassifier(), param_distributions={
                    'leaf_size': list(range(10, 51)),
                    'n_neighbors': list(range(1, 21)),
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                }, n_iter=10, cv=5, n_jobs=-1, verbose=2, refit=True).fit(X_train, y_train)
                st.session_state.leaf_size_knn_b = best.best_params_['leaf_size']
                st.session_state.n_neighbors_knn_b = best.best_params_['n_neighbors']
                st.session_state.weights_knn_b = best.best_params_['weights']
                st.session_state.algorithm_knn_b = best.best_params_['algorithm']

        leaf_size_knn = st.slider("Tamanho da Folha", 10, 50, st.session_state.leaf_size_knn_b, key="leaf_size_knn")
        n_neighbors_knn = st.slider("Número de Vizinhos (K)", 1, 20, st.session_state.n_neighbors_knn_b, key="n_neighbors_knn")
        weights_knn = st.selectbox("Pesos", ['uniform', 'distance'], key="weights_knn", index=['uniform', 'distance'].index(st.session_state.weights_knn_b))
        algorithm_knn = st.selectbox("Algoritmo", ['auto', 'ball_tree', 'kd_tree', 'brute'], key="algorithm_knn", index=['auto', 'ball_tree', 'kd_tree', 'brute'].index(st.session_state.algorithm_knn_b))

    return n_neighbors_knn, weights_knn, algorithm_knn, leaf_size_knn

def parametros_xgboost(st, X_train=None, y_train=None):
    # Definir as chaves de estado da sessão para cada hiperparâmetro
    if 'n_estimators_xgb_g' not in st.session_state:
        st.session_state.n_estimators_xgb_g = 300
    if 'learning_rate_xgb_g' not in st.session_state:
        st.session_state.learning_rate_xgb_g = 0.08
    if 'max_depth_xgb_g' not in st.session_state:
        st.session_state.max_depth_xgb_g = 4
    if 'colsample_bytree_xgb_g' not in st.session_state:
        st.session_state.colsample_bytree_xgb_g = 0.90
    if 'subsample_xgb_g' not in st.session_state:
        st.session_state.subsample_xgb_g = 0.80

    with st.sidebar.expander("Hiperparâmetros - XGBoost", expanded=False):
        if st.button(":rocket: RandomizedSearchCV (Demorado)", key="grid_xgb"):
            with st.spinner("..."):
                best = RandomizedSearchCV(XGBClassifier(random_state=42), param_distributions={
                    'n_estimators': [50, 100, 200, 300, 400, 500],
                    'learning_rate': [0.01, 0.05, 0.08, 0.1, 0.2, 0.3],
                    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'colsample_bytree': [0.1, 0.25, 0.5, 0.75, 0.90, 1.0],
                    'subsample': [0.1, 0.25, 0.5, 0.75, 0.90, 1.0]
                }, n_iter=30, cv=5, n_jobs=-1, verbose=2, refit=True).fit(X_train, y_train)
                st.session_state.n_estimators_xgb_g = best.best_params_['n_estimators']
                st.session_state.learning_rate_xgb_g = best.best_params_['learning_rate']
                st.session_state.max_depth_xgb_g = best.best_params_['max_depth']
                st.session_state.colsample_bytree_xgb_g = best.best_params_['colsample_bytree']
                st.session_state.subsample_xgb_g = best.best_params_['subsample']

        n_estimators_xgb = st.slider("Número de Árvores (Estimadores)", 50, 500, st.session_state.n_estimators_xgb_g, key="n_estimators_xgb")
        learning_rate_xgb = st.number_input("Taxa de Aprendizado", 0.01, 0.5, st.session_state.learning_rate_xgb_g, key="learning_rate_xgb")
        max_depth_xgb = st.slider("Profundidade Máxima", 1, 15, st.session_state.max_depth_xgb_g, key="max_depth_xgb")
        colsample_bytree_xgb = st.number_input("Amostragem de Colunas por Árvore", 0.1, 1.0, st.session_state.colsample_bytree_xgb_g, key="colsample_bytree_xgb")
        subsample_xgb = st.number_input("Amostragem de Filas", 0.1, 1.0, st.session_state.subsample_xgb_g, key="subsample_xgb")

    return n_estimators_xgb, learning_rate_xgb, max_depth_xgb, colsample_bytree_xgb, subsample_xgb