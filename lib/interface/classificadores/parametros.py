def parametros_arvore_decisao(st):
    # Parâmetros para Árvores de Decisão
    with st.sidebar.expander("Hiperparâmetros - Árvores de Decisão",expanded=False):
        max_depth_tree = st.slider("Profundidade Máxima", 1, 20, 10, key="max_depth_tree")
        min_samples_split_tree = st.slider("Mínimo de Amostras para Split", 2, 20, 5, key="min_samples_split_tree")
        min_samples_leaf_tree = st.slider("Mínimo de Amostras na Folha", 1, 10, 2, key="min_samples_leaf_tree")
        max_features_tree = st.selectbox("Máximo de Features", ['sqrt', 'log2', None], key="max_features_tree")
        criterion_tree = st.selectbox("Critério", ["gini", "entropy","log_loss"], index=0, key="criterion_tree")
    
    return max_depth_tree, min_samples_split_tree, min_samples_leaf_tree, max_features_tree, criterion_tree


def parametros_svm(st):
    # Parâmetros para SVM
    with st.sidebar.expander("Hiperparâmetros - SVM",expanded=False):
        C_svm = st.number_input("C (Regularização)", 0.01, 10.0, 0.08, key="C_svm")
        kernel_svm = st.selectbox("Kernel", ['linear', 'poly', 'rbf', 'sigmoid'], key="kernel_svm", index=3)
        degree_svm = st.slider("Grau (se kernel for poly)", 2, 5, 3, key="degree_svm")
        gamma_svm = st.selectbox("Gamma", ['scale', 'auto'], key="gamma_svm")
        coef0_svm = st.slider("Coef0 (para poly/sigmoid)", 0.0, 1.0, 0.50, key="coef0_svm")

    return C_svm, kernel_svm, degree_svm, gamma_svm, coef0_svm

def parametros_random_forest(st):
    # Parâmetros para Random Forest
    with st.sidebar.expander("Hiperparâmetros - Random Forest",expanded=False):
        n_estimators_rf = st.slider("Número de Árvores (Estimadores)", 10, 200, 100, key="n_estimators_rf")
        max_depth_rf = st.slider("Profundidade Máxima", 1, 20, 10, key="max_depth_rf")
        min_samples_split_rf = st.slider("Mínimo de Amostras para Split", 2, 20, 2, key="min_samples_split_rf")
        min_samples_leaf_rf = st.slider("Mínimo de Amostras na Folha", 1, 10, 1, key="min_samples_leaf_rf")
        max_features_rf = st.selectbox("Máximo de Features", ['sqrt', 'log2', None], key="max_features_rf")
        criterion_rf = st.selectbox("Critério", ["gini", "entropy","log_loss"], key="criterion_rf")
        bootstrap_rf = st.selectbox("Bootstrap", [True, False], key="bootstrap_rf")
    
    return n_estimators_rf, max_depth_rf, min_samples_split_rf, min_samples_leaf_rf, max_features_rf, criterion_rf, bootstrap_rf

def parametros_logistic_regression(st):
    # Parâmetros para Logistic Regression
    with st.sidebar.expander("Hiperparâmetros - Logistic Regression", expanded=False):
        C_lr = st.number_input("C (Regularização Inversa)", 0.01, 10.0, 0.01, key="C_lr")
        solver_lr = st.selectbox("Solver", ['liblinear', 'lbfgs', 'sag', 'saga', 'newton-cg'], key="solver_lr")

    return C_lr, solver_lr

def parametros_knn(st):
    # Parâmetros para K-Nearest Neighbors
    with st.sidebar.expander("Hiperparâmetros - KNN", expanded=False):
        n_neighbors_knn = st.slider("Número de Vizinhos (K)", 1, 20, 20, key="n_neighbors_knn")
        weights_knn = st.selectbox("Pesos", ['uniform', 'distance'], key="weights_knn", index=1)
        algorithm_knn = st.selectbox("Algoritmo", ['auto', 'ball_tree', 'kd_tree', 'brute'], key="algorithm_knn", index=3)

    return n_neighbors_knn, weights_knn, algorithm_knn

def parametros_xgboost(st):
    # Parâmetros para XGBoost
    with st.sidebar.expander("Hiperparâmetros - XGBoost", expanded=False):
        n_estimators_xgb = st.slider("Número de Árvores (Estimadores)", 50, 500, 100, key="n_estimators_xgb")
        learning_rate_xgb = st.number_input("Taxa de Aprendizado", 0.01, 0.5, 0.25, key="learning_rate_xgb")
        max_depth_xgb = st.slider("Profundidade Máxima", 1, 15, 4, key="max_depth_xgb")
        colsample_bytree_xgb = st.number_input("Amostragem de Colunas por Árvore", 0.1, 1.0, 0.90, key="colsample_bytree_xgb")
        subsample_xgb = st.number_input("Amostragem de Filas", 0.1, 1.0, 0.8, key="subsample_xgb")

    return n_estimators_xgb, learning_rate_xgb, max_depth_xgb, colsample_bytree_xgb, subsample_xgb
