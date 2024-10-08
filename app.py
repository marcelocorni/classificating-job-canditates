from sklearn.discriminant_analysis import StandardScaler
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from lib.interface.cabecalhos import cria_cabecalho_padrao
from lib.interface.classificadores.parametros import parametros_arvore_decisao, parametros_knn, parametros_logistic_regression, parametros_random_forest, parametros_svm, parametros_xgboost
from lib.interface.coloracao import apply_styles, color_metric, styled_metric_text, styled_metric_text_classes
from lib.treinamento.relatorios import classification_report_to_df
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import Lasso

def main():
    st.set_page_config(page_title="Trabalho 5 - Clasificação", layout="wide")

    pl.style.use('dark_background')

    cria_cabecalho_padrao(st,'Classificação dos Dados Agrupados de Candidatos a Emprego',
                        'Análise Classificativa de Dados Agrupados',
                        'Mineração de Dados',
                        'Agosto/2024')


    @st.cache_data
    def treinar_arvore_decisao(X_train, y_train, max_depth_tree, min_samples_split_tree, min_samples_leaf_tree, max_features_tree, criterion_tree, ccp_alpha):
        clf_tree = DecisionTreeClassifier(max_depth=max_depth_tree,
                                    min_samples_split=min_samples_split_tree,
                                    min_samples_leaf=min_samples_leaf_tree,
                                    max_features=max_features_tree,
                                    criterion=criterion_tree,
                                    ccp_alpha=ccp_alpha,
                                    random_state=42)
        clf_tree.fit(X_train, y_train)
        return clf_tree


    @st.cache_data
    def treinar_svm(X_train, y_train, C_svm, kernel_svm, degree_svm, gamma_svm, coef0_svm):
        clf_svm = SVC(C=C_svm, kernel=kernel_svm, degree=degree_svm, gamma=gamma_svm, coef0=coef0_svm, random_state=42)
        clf_svm.fit(X_train, y_train)
        return clf_svm


    @st.cache_data
    def treinar_random_forest(X_train, y_train, n_estimators_rf, max_depth_rf, min_samples_split_rf, min_samples_leaf_rf, max_features_rf, criterion_rf, bootstrap_rf):
        clf_rf = RandomForestClassifier(n_estimators=n_estimators_rf,
                                    max_depth=max_depth_rf,
                                    min_samples_split=min_samples_split_rf,
                                    min_samples_leaf=min_samples_leaf_rf,
                                    max_features=max_features_rf,
                                    bootstrap=bootstrap_rf,
                                    criterion=criterion_rf,
                                    random_state=42)
        clf_rf.fit(X_train, y_train)
        return clf_rf


    @st.cache_data
    def treinar_logistic_regression(X_train, y_train, C_lr, solver_lr):
        clf_lr = LogisticRegression(C=C_lr, solver=solver_lr, random_state=42)
        clf_lr.fit(X_train, y_train)
        return clf_lr

    @st.cache_data
    def treinar_knn(X_train, y_train, n_neighbors_knn, weights_knn, algorithm_knn, leaf_size_knn):
        clf_knn = KNeighborsClassifier(n_jobs=-1,leaf_size=leaf_size_knn, n_neighbors=n_neighbors_knn, weights=weights_knn, algorithm=algorithm_knn)
        clf_knn.fit(X_train, y_train)
        return clf_knn

    @st.cache_data
    def treinar_xgboost(X_train, y_train, n_estimators_xgb, learning_rate_xgb, max_depth_xgb, colsample_bytree_xgb,subsample_xgb):
        clf_xgb = XGBClassifier(n_estimators=n_estimators_xgb,
                            learning_rate=learning_rate_xgb,
                            max_depth=max_depth_xgb,
                            colsample_bytree=colsample_bytree_xgb,
                            subsample=subsample_xgb,
                            random_state=42)

        clf_xgb.fit(X_train, y_train)
        return clf_xgb


    @st.cache_data
    def treinar_lasso(data, X, y):
        # Treinar modelo Lasso
        lasso = Lasso(alpha=0.1)
        lasso.fit(X, y)
        # Coeficientes das features
        importance = np.abs(lasso.coef_)
        selected_features = np.array(X.columns)[importance > 0]
        data = data[selected_features]
        return data

    test_size = st.slider("Proporção de Teste", 0.1, 0.5, 0.28, 0.01)


    # Carregar os dados
    data = pd.read_csv('data/dados_normalizados.csv')

    # Adicionar nova coluna label para classificação BP ou NAO_BP
    data['label'] = data['performance'].apply(lambda x: 1 if x == 'BP' else 0)

    col1, col2 = st.columns(2)


    # Remover a coluna 'performance' do DataFrame para não ser incluída no playground
    X = data.drop(columns=['performance', 'label'])
    y = data['label']

    # Treinar o modelo Lasso para selecionar as features
    if st.checkbox("Treinar Lasso para Seleção de Features",value=False, key="treinar_lasso"):
        with st.spinner("..."):
            dados = treinar_lasso(data, X, y)
            # Atualizar X
            X = dados
   
    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Exibir a disribuição das classes
    with col1:
        st.write(f"###### Distribuição das Classes antes do balanceamento")
        st.write(y_train.value_counts().reset_index().rename(columns={'index': 'Classe', 'label': 'Contagem'}))

    under_over_sampling = st.selectbox("Escolha a técnica de balanceamento", ['Random Under Sampler', 'SMOTE'], key="sampling", index=1)

    if under_over_sampling == 'Random Under Sampler':
        # Igualar a quantidade de amostras de cada classe
        under = RandomUnderSampler(random_state=42)
        X_train, y_train = under.fit_resample(X_train, y_train)
    else:
        # Igualar a quantidade de amostras de cada classe
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # Exibindo a distribuição das classes após o balanceamento com SMOTE
    with col2:    
        st.write(f"###### Distribuição das Classes após aplicação do {under_over_sampling}")
        st.write(y_train.value_counts().reset_index().rename(columns={'index': 'Classe', 'label': 'Contagem'}))

    #Apresentar e retornar os valores dos hiperparâmetros na sidebar
    max_depth_tree, min_samples_split_tree, min_samples_leaf_tree, max_features_tree, criterion_tree, ccp_alpha = parametros_arvore_decisao(st, X_train, y_train)
    C_svm, kernel_svm, degree_svm, gamma_svm, coef0_svm = parametros_svm(st, X_train, y_train)
    n_estimators_rf, max_depth_rf, min_samples_split_rf, min_samples_leaf_rf, max_features_rf, criterion_rf, bootstrap_rf = parametros_random_forest(st, X_train, y_train)
    C_lr, solver_lr = parametros_logistic_regression(st, X_train, y_train)
    n_neighbors_knn, weights_knn, algorithm_knn,leaf_size_knn = parametros_knn(st, X_train, y_train)
    n_estimators_xgb, learning_rate_xgb, max_depth_xgb, colsample_bytree_xgb, subsample_xgb = parametros_xgboost(st, X_train, y_train)

    # Treinar os modelos
    clf_tree = treinar_arvore_decisao(X_train, y_train, max_depth_tree, min_samples_split_tree, min_samples_leaf_tree, max_features_tree, criterion_tree, ccp_alpha)
    clf_svm = treinar_svm(X_train, y_train, C_svm, kernel_svm, degree_svm, gamma_svm, coef0_svm)
    clf_rf = treinar_random_forest(X_train, y_train, n_estimators_rf, max_depth_rf, min_samples_split_rf, min_samples_leaf_rf, max_features_rf, criterion_rf, bootstrap_rf)
    clf_lr = treinar_logistic_regression(X_train, y_train, C_lr, solver_lr)
    clf_knn = treinar_knn(X_train, y_train, n_neighbors_knn, weights_knn, algorithm_knn, leaf_size_knn)
    clf_xgb = treinar_xgboost(X_train, y_train, n_estimators_xgb, learning_rate_xgb, max_depth_xgb, colsample_bytree_xgb, subsample_xgb)

    # Fazer predições
    y_pred_tree = clf_tree.predict(X_test)
    y_pred_svm = clf_svm.predict(X_test)
    y_pred_rf = clf_rf.predict(X_test)
    y_pred_lr = clf_lr.predict(X_test)
    y_pred_knn = clf_knn.predict(X_test)
    y_pred_xgb = clf_xgb.predict(X_test)

    # Converter as predições para as classes nomeadas
    class_mapping_binary = {0: "NAO_BP", 1: "BP"}
    y_pred_rf_mapped = [class_mapping_binary.get(pred, "Desconhecido") for pred in y_pred_rf]

    # Converter as predições para as classes nomeadas
    y_pred_lr_mapped = [class_mapping_binary.get(pred, "Desconhecido") for pred in y_pred_lr]
    y_pred_knn_mapped = [class_mapping_binary.get(pred, "Desconhecido") for pred in y_pred_knn]

    # Converter as predições para as classes nomeadas
    y_pred_xgb_mapped = [class_mapping_binary.get(pred, "Desconhecido") for pred in y_pred_xgb]


    # Criar gráficos de dispersão 2D para visualizar as classificações
    pca = PCA(n_components=2)
    X_test_pca = pca.fit_transform(X_test)

    # Converter as predições para as classes nomeadas
    y_pred_tree_mapped = [class_mapping_binary.get(pred, "Desconhecido") for pred in y_pred_tree]
    y_pred_svm_mapped = [class_mapping_binary.get(pred, "Desconhecido") for pred in y_pred_svm]

    # Definir explicitamente as cores para as classes
    color_discrete_map_binary = {
        "BP": "green",
        "NAO_BP": "red"
    }

    # Substituir os números das classes pelos nomes no gráfico de comparação
    comparison_data = pd.DataFrame({
        'Árvores de Decisão': [class_mapping_binary.get(pred, "Desconhecido") for pred in y_pred_tree],
        'SVM': [class_mapping_binary.get(pred, "Desconhecido") for pred in y_pred_svm],
        'Random Forest': [class_mapping_binary.get(pred, "Desconhecido") for pred in y_pred_rf]
    })

    with st.expander("Árvore de Decisão",expanded=False):
        col1, col2 = st.columns(2)
        report_tree = classification_report(y_test, y_pred_tree)
        df_tree = classification_report_to_df(report_tree)
        accuracy_tree = accuracy_score(y_test, y_pred_tree)
        precision_tree = precision_score(y_test, y_pred_tree, average='weighted')
        recall_tree = recall_score(y_test, y_pred_tree, average='weighted')
        f1_tree = f1_score(y_test, y_pred_tree, average='weighted')
        df_tree_styled = df_tree.style.applymap(color_metric, subset=['precision', 'recall', 'f1-score'])
        with col1:
            st.subheader("Resultados")
            st.dataframe(df_tree_styled)
            st.write((f'Acurácia: {styled_metric_text(accuracy_tree,'')}'), unsafe_allow_html=True)
            st.write((f'Precisão: {styled_metric_text(precision_tree,'')}'), unsafe_allow_html=True)
            st.write((f'Recall: {styled_metric_text(recall_tree,'')}'), unsafe_allow_html=True)
            st.write((f'F1-Score: {styled_metric_text(f1_tree,'')}'), unsafe_allow_html=True)

            st.write("### Parâmetros do Modelo")
            st.write(clf_tree.get_params())

            # Obter a importância das features
            importances_tree = clf_tree.feature_importances_
            indices_tree = np.argsort(importances_tree)[::-1]  # Ordenar as features pela importância

            # Criar um DataFrame para exibir as features com sua importância
            feature_importance_df_tree = pd.DataFrame({
                'Feature': X.columns[indices_tree],
                'Importância': importances_tree[indices_tree]
            })

            # Exibir o DataFrame de importância das features
            st.write("### Importância das Features")
            st.dataframe(feature_importance_df_tree)

            # Exibindo o gráfico da matriz de confusão
            cm_tree = confusion_matrix(y_test, y_pred_tree)
            
            # Criar o gráfico de matriz de confusão
            fig = go.Figure(data=go.Heatmap(
                z=cm_tree,
                x=['Previsto NAO_BP', 'Previsto BP'],
                y=['Real NAO_BP', 'Real BP'],
                colorscale='tealgrn',
                colorbar=dict(title='Count'),
                zmin=0,
                zmax=np.max(cm_tree),
                texttemplate='%{z}',
                textfont=dict(size=14, color='white'),
                showscale=True
            ))

            fig.update_layout(
                title='Matriz de Confusão - Árvore de Decisão',
                xaxis_title='Predito',
                yaxis_title='Real',
                xaxis=dict(tickvals=[0, 1], ticktext=['NAO_BP', 'BP']),
                yaxis=dict(tickvals=[0, 1], ticktext=['NAO_BP', 'BP']),
                width=600,
                height=500
            )

            st.plotly_chart(fig)

        with col2:
            fig_tree = px.scatter(x=X_test_pca[:, 0], y=X_test_pca[:, 1], color=y_pred_tree_mapped,
                                labels={'x': 'PC1', 'y': 'PC2', 'color': 'Classe Predita'},
                                title="Árvores de Decisão - Dispersão das Classes Preditas",
                                color_discrete_map=color_discrete_map_binary)

            st.plotly_chart(fig_tree)

            fig_tree_dist = px.histogram(
                x=y_pred_tree_mapped,
                labels={'x': 'Classe Predita', 'y': 'Contagem'},
                title="Árvores de Decisão - Distribuição das Classes Preditas",
                color_discrete_map=color_discrete_map_binary
            )

            st.plotly_chart(fig_tree_dist)

            # Gráfico de barras para visualização
            fig, ax = pl.subplots(figsize=(10, 6))
            ax.barh(feature_importance_df_tree['Feature'], feature_importance_df_tree['Importância'], color='teal')
            ax.set_xlabel('Importância')
            ax.set_title('Importância das Features na Árvore de Decisão')
            ax.invert_yaxis()  # Para que a feature mais importante apareça no topo
            st.pyplot(fig)

        
        max_depth_visualizacao = st.slider(
            "Escolha a Profundidade Máxima da Árvore para Visualização", 
            0, 
            clf_tree.get_depth(), 
            clf_tree.get_depth()
        )

        # Plotar a árvore até o max_depth especificado
        fig, ax = pl.subplots(figsize=(20, 10))

        # plot_tree aceita max_depth mas não min_depth, então você verá a árvore a partir da raiz até o max_depth.
        plot_tree(clf_tree, 
                max_depth=max_depth_visualizacao, 
                filled=False, 
                feature_names=X.columns, 
                class_names=list(class_mapping_binary.values()), 
                ax=ax)

        pl.title(f"Árvore de Decisão (Níveis: {max_depth_visualizacao})")
        st.pyplot(fig)

    with st.expander("SVM",expanded=False):
        col1, col2 = st.columns(2)
        report_svm = classification_report(y_test, y_pred_svm)
        df_svm = classification_report_to_df(report_svm)
        accuracy_svm = accuracy_score(y_test, y_pred_svm)
        precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
        recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
        f1_svm = f1_score(y_test, y_pred_svm, average='weighted')
        df_svm_styled = df_svm.style.applymap(color_metric, subset=['precision', 'recall', 'f1-score'])

        with col1:
            st.subheader("Resultados")
            st.dataframe(df_svm_styled)
            st.write((f'Acurácia: {styled_metric_text(accuracy_svm,'')}'), unsafe_allow_html=True)
            st.write((f'Precisão: {styled_metric_text(precision_svm,'')}'), unsafe_allow_html=True)
            st.write((f'Recall: {styled_metric_text(recall_svm,'')}'), unsafe_allow_html=True)
            st.write((f'F1-Score: {styled_metric_text(f1_svm,'')}'), unsafe_allow_html=True)

            st.write("### Parâmetros do Modelo")
            st.write(clf_svm.get_params())

            if kernel_svm == 'linear':
                # Análise para SVM Linear usando coeficientes
                st.write("### Importância das Features no SVM (Kernel Linear)")

                # Obter os coeficientes das features
                coef = clf_svm.coef_.flatten()  # Achatar o array para facilitar a manipulação
                indices = np.argsort(np.abs(coef))[::-1]  # Ordenar pela importância absoluta dos coeficientes

                # Criar um DataFrame para exibir as features com sua importância
                feature_importance_df = pd.DataFrame({
                    'Feature': X.columns[indices],
                    'Coeficiente': coef[indices]
                })

                # Exibir o DataFrame de importância das features
                st.write("#### Coeficientes das Features (SVM Linear)")
                st.dataframe(feature_importance_df)
            else:
                st.write("### Importância das Features não disponível para SVM com Kernel não-linear")
            
            # Exibindo o gráfico da matriz de confusão
            cm_smv = confusion_matrix(y_test, y_pred_svm)
            
            # Criar o gráfico de matriz de confusão
            fig_cm_svm = go.Figure(data=go.Heatmap(
                z=cm_smv,
                x=['Previsto NAO_BP', 'Previsto BP'],
                y=['Real NAO_BP', 'Real BP'],
                colorscale='tealgrn',
                colorbar=dict(title='Count'),
                zmin=0,
                zmax=np.max(cm_smv),
                texttemplate='%{z}',
                textfont=dict(size=14, color='white'),
                showscale=True
            ))

            fig_cm_svm.update_layout(
                title='Matriz de Confusão - SVM',
                xaxis_title='Predito',
                yaxis_title='Real',
                xaxis=dict(tickvals=[0, 1], ticktext=['NAO_BP', 'BP']),
                yaxis=dict(tickvals=[0, 1], ticktext=['NAO_BP', 'BP']),
                width=600,
                height=500
            )

            st.plotly_chart(fig_cm_svm)

        with col2:
            fig_tree.for_each_trace(lambda t: t.update(name=class_mapping_binary[int(t.name)]) if t.name.isdigit() else None)
            fig_svm = px.scatter(x=X_test_pca[:, 0], y=X_test_pca[:, 1], color=y_pred_svm_mapped,
                                labels={'x': 'PC1', 'y': 'PC2', 'color': 'Classe Predita'},
                                title="SVM - Dispersão das Classes Preditas",
                                color_discrete_map=color_discrete_map_binary)

            st.plotly_chart(fig_svm)

            fig_svm_dist = px.histogram(
                x=y_pred_svm_mapped,
                labels={'x': 'Classe Predita', 'y': 'Contagem'},
                title="SVM - Distribuição das Classes Preditas"
            )
            st.plotly_chart(fig_svm_dist)

            if kernel_svm == 'linear':
                # Gráfico de barras para visualização
                fig, ax = pl.subplots(figsize=(10, 6))
                ax.barh(feature_importance_df['Feature'], feature_importance_df['Coeficiente'], color='purple')
                ax.set_xlabel('Coeficiente')
                ax.set_title('Coeficientes das Features no SVM Linear')
                ax.invert_yaxis()  # Para que a feature mais importante apareça no topo
                col2.pyplot(fig)

    with st.expander("Random Forest", expanded=False):
        col1, col2 = st.columns(2)
        report_rf = classification_report(y_test, y_pred_rf)
        df_rf = classification_report_to_df(report_rf)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
        recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
        f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
        df_rf_styled = df_rf.style.applymap(color_metric, subset=['precision', 'recall', 'f1-score'])
        
        with col1:
            st.subheader("Resultados")
            st.dataframe(df_rf_styled)
            st.write(f'Acurácia: {styled_metric_text(accuracy_rf, "")}', unsafe_allow_html=True)
            st.write(f'Precisão: {styled_metric_text(precision_rf, "")}', unsafe_allow_html=True)
            st.write(f'Recall: {styled_metric_text(recall_rf, "")}', unsafe_allow_html=True)
            st.write(f'F1-Score: {styled_metric_text(f1_rf, "")}', unsafe_allow_html=True)

            st.write("### Parâmetros do Modelo")
            st.write(clf_rf.get_params())

            # Obter a importância das features
            importances = clf_rf.feature_importances_
            indices = np.argsort(importances)[::-1]  # Ordenar as features pela importância

            # Criar um DataFrame para exibir as features com sua importância
            feature_importance_df = pd.DataFrame({
                'Feature': X.columns[indices],
                'Importância': importances[indices]
            })

            # Exibir o DataFrame de importância das features
            st.write("### Importância das Features")
            st.dataframe(feature_importance_df)

            # Exibindo o gráfico da matriz de confusão
            cm_rf = confusion_matrix(y_test, y_pred_rf)

            # Criar o gráfico de matriz de confusão
            fig_cm_rf = go.Figure(data=go.Heatmap(
                z=cm_rf,
                x=['Previsto NAO_BP', 'Previsto BP'],
                y=['Real NAO_BP', 'Real BP'],
                colorscale='tealgrn',
                colorbar=dict(title='Count'),
                zmin=0,
                zmax=np.max(cm_rf),
                texttemplate='%{z}',
                textfont=dict(size=14, color='white'),
                showscale=True
            ))

            fig_cm_rf.update_layout(
                title='Matriz de Confusão - Random Forest',
                xaxis_title='Predito',
                yaxis_title='Real',
                xaxis=dict(tickvals=[0, 1], ticktext=['NAO_BP', 'BP']),
                yaxis=dict(tickvals=[0, 1], ticktext=['NAO_BP', 'BP']),
                width=600,
                height=500
            )

            st.plotly_chart(fig_cm_rf)

        with col2:
            fig_rf = px.scatter(x=X_test_pca[:, 0], y=X_test_pca[:, 1], color=y_pred_rf_mapped,
                                labels={'x': 'PC1', 'y': 'PC2', 'color': 'Classe Predita'},
                                title="Random Forest - Dispersão das Classes Preditas",
                                color_discrete_map=color_discrete_map_binary)
            
            st.plotly_chart(fig_rf)

            # Random Forest
            fig_rf_dist = px.histogram(
                x=y_pred_rf_mapped,
                labels={'x': 'Classe Predita', 'y': 'Contagem'},
                title="Random Forest - Distribuição das Classes Preditas"
            )
            st.plotly_chart(fig_rf_dist)

            # Gráfico de barras para visualização
            fig, ax = pl.subplots(figsize=(10, 6))
            ax.barh(feature_importance_df['Feature'], feature_importance_df['Importância'], color='teal')
            ax.set_xlabel('Importância')
            ax.set_title('Importância das Features no Random Forest')
            ax.invert_yaxis()  # Para que a feature mais importante apareça no topo
            st.pyplot(fig)

        # Adicionar uma opção para selecionar a árvore de um Random Forest
        if isinstance(clf_rf, RandomForestClassifier):
            tree_index = st.selectbox("Selecione a Árvore para Visualizar", range(len(clf_rf.estimators_)))

            # Visualizar a árvore selecionada
            estimator = clf_rf.estimators_[tree_index]
            st.subheader(f"Visualização da Árvore {tree_index + 1} da Random Forest")


            max_depth_visualizacao_rf = st.slider(
                    "Escolha a Profundidade Máxima da Árvore para Visualização", 
                    0, 
                    estimator.get_depth(),
                    estimator.get_depth(),
                    key="max_depth_visualizacao_rf"
                )
            
            fig, ax = pl.subplots(figsize=(20, 10))
            plot_tree(estimator, 
                max_depth=max_depth_visualizacao_rf, 
                filled=False, 
                feature_names=X.columns, 
                class_names=list(class_mapping_binary.values()), 
                ax=ax)

            pl.title(f"Árvore {tree_index + 1} da Random Forest")
            st.pyplot(fig)


    # Logistic Regression
    with st.expander("Logistic Regression", expanded=False):
        col1, col2 = st.columns(2)
        report_lr = classification_report(y_test, y_pred_lr)
        df_lr = classification_report_to_df(report_lr)
        accuracy_lr = accuracy_score(y_test, y_pred_lr)
        precision_lr = precision_score(y_test, y_pred_lr, average='weighted')
        recall_lr = recall_score(y_test, y_pred_lr, average='weighted')
        f1_lr = f1_score(y_test, y_pred_lr, average='weighted')
        df_lr_styled = df_lr.style.applymap(color_metric, subset=['precision', 'recall', 'f1-score'])
        
        with col1:
            st.subheader("Resultados")
            st.dataframe(df_lr_styled)
            st.write(f'Acurácia: {styled_metric_text(accuracy_lr, "")}', unsafe_allow_html=True)
            st.write(f'Precisão: {styled_metric_text(precision_lr, "")}', unsafe_allow_html=True)
            st.write(f'Recall: {styled_metric_text(recall_lr, "")}', unsafe_allow_html=True)
            st.write(f'F1-Score: {styled_metric_text(f1_lr, "")}', unsafe_allow_html=True)

            st.write("### Parâmetros do Modelo")
            st.write(clf_lr.get_params())

            # Importância das Features usando Coeficientes
            coef_lr = clf_lr.coef_.flatten()
            indices_lr = np.argsort(np.abs(coef_lr))[::-1]  # Ordenar pela importância absoluta dos coeficientes

            # Criar DataFrame para exibir as features com sua importância
            feature_importance_df_lr = pd.DataFrame({
                'Feature': X.columns[indices_lr],
                'Coeficiente': coef_lr[indices_lr]
            })

            # Exibir o DataFrame de importância das features
            st.write("### Importância das Features (Coeficientes)")
            st.dataframe(feature_importance_df_lr)

            # Exibindo o gráfico da matriz de confusão
            cm_lr = confusion_matrix(y_test, y_pred_lr)

            # Criar o gráfico de matriz de confusão
            fig_cm_lr = go.Figure(data=go.Heatmap(
                z=cm_lr,
                x=['Previsto NAO_BP', 'Previsto BP'],
                y=['Real NAO_BP', 'Real BP'],
                colorscale='tealgrn',
                colorbar=dict(title='Count'),
                zmin=0,
                zmax=np.max(cm_lr),
                texttemplate='%{z}',
                textfont=dict(size=14, color='white'),
                showscale=True
            ))

            fig_cm_lr.update_layout(
                title='Matriz de Confusão - Logistic Regression',
                xaxis_title='Predito',
                yaxis_title='Real',
                xaxis=dict(tickvals=[0, 1], ticktext=['NAO_BP', 'BP']),
                yaxis=dict(tickvals=[0, 1], ticktext=['NAO_BP', 'BP']),
                width=600,
                height=500
            )

            st.plotly_chart(fig_cm_lr)

        with col2:
            fig_lr = px.scatter(x=X_test_pca[:, 0], y=X_test_pca[:, 1], color=y_pred_lr_mapped,
                                labels={'x': 'PC1', 'y': 'PC2', 'color': 'Classe Predita'},
                                title="Logistic Regression - Dispersão das Classes Preditas",
                                color_discrete_map=color_discrete_map_binary)
            
            st.plotly_chart(fig_lr)

            fig_lr_dist = px.histogram(
                x=y_pred_lr_mapped,
                labels={'x': 'Classe Predita', 'y': 'Contagem'},
                title="Logistic Regression - Distribuição das Classes Preditas"
            )
            st.plotly_chart(fig_lr_dist)

            # Gráfico de barras para visualização
            fig, ax = pl.subplots(figsize=(10, 6))
            ax.barh(feature_importance_df_lr['Feature'], feature_importance_df_lr['Coeficiente'], color='blue')
            ax.set_xlabel('Coeficiente')
            ax.set_title('Importância das Features na Logistic Regression')
            ax.invert_yaxis()  # Para que a feature mais importante apareça no topo
            st.pyplot(fig)

    # K-Nearest Neighbors
    with st.expander("K-Nearest Neighbors", expanded=False):
        col1, col2 = st.columns(2)
        report_knn = classification_report(y_test, y_pred_knn)
        df_knn = classification_report_to_df(report_knn)
        accuracy_knn = accuracy_score(y_test, y_pred_knn)
        precision_knn = precision_score(y_test, y_pred_knn, average='weighted')
        recall_knn = recall_score(y_test, y_pred_knn, average='weighted')
        f1_knn = f1_score(y_test, y_pred_knn, average='weighted')
        df_knn_styled = df_knn.style.applymap(color_metric, subset=['precision', 'recall', 'f1-score'])

        with col1:
            st.subheader("Resultados")
            st.dataframe(df_knn_styled)
            st.write(f'Acurácia: {styled_metric_text(accuracy_knn, "")}', unsafe_allow_html=True)
            st.write(f'Precisão: {styled_metric_text(precision_knn, "")}', unsafe_allow_html=True)
            st.write(f'Recall: {styled_metric_text(recall_knn, "")}', unsafe_allow_html=True)
            st.write(f'F1-Score: {styled_metric_text(f1_knn, "")}', unsafe_allow_html=True)

            st.write("### Parâmetros do Modelo")
            st.write(clf_knn.get_params())

            # Exibindo o gráfico da matriz de confusão
            cm_knn = confusion_matrix(y_test, y_pred_knn)

            # Criar o gráfico de matriz de confusão
            fig_cm_knn = go.Figure(data=go.Heatmap(
                z=cm_knn,
                x=['Previsto NAO_BP', 'Previsto BP'],
                y=['Real NAO_BP', 'Real BP'],
                colorscale='tealgrn',
                colorbar=dict(title='Count'),
                zmin=0,
                zmax=np.max(cm_knn),
                texttemplate='%{z}',
                textfont=dict(size=14, color='white'),
                showscale=True
            ))

            fig_cm_knn.update_layout(
                title='Matriz de Confusão - K-Nearest Neighbors',
                xaxis_title='Predito',
                yaxis_title='Real',
                xaxis=dict(tickvals=[0, 1], ticktext=['NAO_BP', 'BP']),
                yaxis=dict(tickvals=[0, 1], ticktext=['NAO_BP', 'BP']),
                width=600,
                height=500
            )

            st.plotly_chart(fig_cm_knn)

        with col2:
            fig_knn = px.scatter(x=X_test_pca[:, 0], y=X_test_pca[:, 1], color=y_pred_knn_mapped,
                                labels={'x': 'PC1', 'y': 'PC2', 'color': 'Classe Predita'},
                                title="K-Nearest Neighbors - Dispersão das Classes Preditas",
                                color_discrete_map=color_discrete_map_binary)
            
            st.plotly_chart(fig_knn)

            fig_knn_dist = px.histogram(
                x=y_pred_knn_mapped,
                labels={'x': 'Classe Predita', 'y': 'Contagem'},
                title="K-Nearest Neighbors - Distribuição das Classes Preditas"
            )
            st.plotly_chart(fig_knn_dist)

    # XGBoost
    with st.expander("XGBoost", expanded=False):
        col1, col2 = st.columns(2)
        report_xgb = classification_report(y_test, y_pred_xgb)
        df_xgb = classification_report_to_df(report_xgb)
        accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
        precision_xgb = precision_score(y_test, y_pred_xgb, average='weighted')
        recall_xgb = recall_score(y_test, y_pred_xgb, average='weighted')
        f1_xgb = f1_score(y_test, y_pred_xgb, average='weighted')
        df_xgb_styled = df_xgb.style.applymap(color_metric, subset=['precision', 'recall', 'f1-score'])

        with col1:
            st.subheader("Resultados")
            st.dataframe(df_xgb_styled)
            st.write(f'Acurácia: {styled_metric_text(accuracy_xgb, "")}', unsafe_allow_html=True)
            st.write(f'Precisão: {styled_metric_text(precision_xgb, "")}', unsafe_allow_html=True)
            st.write(f'Recall: {styled_metric_text(recall_xgb, "")}', unsafe_allow_html=True)
            st.write(f'F1-Score: {styled_metric_text(f1_xgb, "")}', unsafe_allow_html=True)

            st.write("### Parâmetros do Modelo")
            st.write(clf_xgb.get_params())

            # Obter a importância das features
            importances_xgb = clf_xgb.feature_importances_
            indices_xgb = np.argsort(importances_xgb)[::-1]  # Ordenar as features pela importância

            # Criar um DataFrame para exibir as features com sua importância
            feature_importance_df_xgb = pd.DataFrame({
                'Feature': X.columns[indices_xgb],
                'Importância': importances_xgb[indices_xgb]
            })

            # Exibir o DataFrame de importância das features
            st.write("### Importância das Features")
            st.dataframe(feature_importance_df_xgb)

            # Exibindo o gráfico da matriz de confusão
            cm_xgb = confusion_matrix(y_test, y_pred_xgb)

            # Criar o gráfico de matriz de confusão
            fig_cm_xgb = go.Figure(data=go.Heatmap(
                z=cm_xgb,
                x=['Previsto NAO_BP', 'Previsto BP'],
                y=['Real NAO_BP', 'Real BP'],
                colorscale='tealgrn',
                colorbar=dict(title='Count'),
                zmin=0,
                zmax=np.max(cm_xgb),
                texttemplate='%{z}',
                textfont=dict(size=14, color='white'),
                showscale=True
            ))

            fig_cm_xgb.update_layout(
                title='Matriz de Confusão - XGBoost',
                xaxis_title='Predito',
                yaxis_title='Real',
                xaxis=dict(tickvals=[0, 1], ticktext=['NAO_BP', 'BP']),
                yaxis=dict(tickvals=[0, 1], ticktext=['NAO_BP', 'BP']),
                width=600,
                height=500
            )

            st.plotly_chart(fig_cm_xgb)

        with col2:
            fig_xgb = px.scatter(x=X_test_pca[:, 0], y=X_test_pca[:, 1], color=y_pred_xgb_mapped,
                                labels={'x': 'PC1', 'y': 'PC2', 'color': 'Classe Predita'},
                                title="XGBoost - Dispersão das Classes Preditas",
                                color_discrete_map=color_discrete_map_binary)
            
            st.plotly_chart(fig_xgb)

            fig_xgb_dist = px.histogram(
                x=y_pred_xgb_mapped,
                labels={'x': 'Classe Predita', 'y': 'Contagem'},
                title="XGBoost - Distribuição das Classes Preditas"
            )
            st.plotly_chart(fig_xgb_dist)

            # Gráfico de barras para visualização
            fig, ax = pl.subplots(figsize=(10, 6))
            ax.barh(feature_importance_df_xgb['Feature'], feature_importance_df_xgb['Importância'], color='orange')
            ax.set_xlabel('Importância')
            ax.set_title('Importância das Features no XGBoost')
            ax.invert_yaxis()  # Para que a feature mais importante apareça no topo
            st.pyplot(fig)

    with st.expander("Comparação dos Classificadores", expanded=False):

        #Definir o mapa de cores para os classificadores
        color_discrete_map_classificadores = {
            "Árvores de Decisão": "rgb(124, 108, 119)",
            "SVM": "rgb(170, 166, 148)",
            "Random Forest": "rgb(209, 208, 163)",
            "Logistic Regression": "rgb(235, 248, 184)",
            "KNN": "rgb(255, 231, 135)",
            "XGBoost": "rgb(128, 143, 133)"
        }

        # Atualizar a comparação da distribuição das classes preditas pelos classificadores
        comparison_data = pd.DataFrame({
            'Árvores de Decisão': [class_mapping_binary.get(pred, "Desconhecido") for pred in y_pred_tree],
            'SVM': [class_mapping_binary.get(pred, "Desconhecido") for pred in y_pred_svm],
            'Random Forest': [class_mapping_binary.get(pred, "Desconhecido") for pred in y_pred_rf],
            'Logistic Regression': [class_mapping_binary.get(pred, "Desconhecido") for pred in y_pred_lr],
            'KNN': [class_mapping_binary.get(pred, "Desconhecido") for pred in y_pred_knn],
            'XGBoost': [class_mapping_binary.get(pred, "Desconhecido") for pred in y_pred_xgb]
        })

        # Gráfico de barras empilhadas para comparação das predições
        fig_comparison = px.histogram(comparison_data, barmode='group', 
                                    labels={'value': 'Classe Predita', 'variable': 'Classificador'},
                                    title="Distribuição das Classes Preditas por Diferentes Classificadores",
                                    color_discrete_map=color_discrete_map_classificadores)
        st.plotly_chart(fig_comparison)

        col1, col2 = st.columns(2)
        with col1:
            # Tabela de comparação das previsões
            comparison_df = pd.DataFrame({
                'Real': [class_mapping_binary.get(real, "Desconhecido") for real in y_test],
                'Árvores de Decisão': [class_mapping_binary.get(pred, "Desconhecido") for pred in y_pred_tree],
                'SVM': [class_mapping_binary.get(pred, "Desconhecido") for pred in y_pred_svm],
                'Random Forest': [class_mapping_binary.get(pred, "Desconhecido") for pred in y_pred_rf],
                'Logistic Regression': [class_mapping_binary.get(pred, "Desconhecido") for pred in y_pred_lr],
                'KNN': [class_mapping_binary.get(pred, "Desconhecido") for pred in y_pred_knn],
                'XGBoost': [class_mapping_binary.get(pred, "Desconhecido") for pred in y_pred_xgb]
            })
            
            # Aplicar os estilos ao DataFrame
            comparison_styled = comparison_df.style.apply(apply_styles, axis=1)
            
            st.write("### Tabela de Comparação das Previsões")
            st.dataframe(comparison_styled)

        with col2:
            # Cálculo do percentual de erro para cada classificador
            total_samples = len(y_test)
            error_tree = sum(comparison_df['Árvores de Decisão'] != comparison_df['Real']) / total_samples * 100
            error_svm = sum(comparison_df['SVM'] != comparison_df['Real']) / total_samples * 100
            error_rf = sum(comparison_df['Random Forest'] != comparison_df['Real']) / total_samples * 100
            error_lr = sum(comparison_df['Logistic Regression'] != comparison_df['Real']) / total_samples * 100
            error_knn = sum(comparison_df['KNN'] != comparison_df['Real']) / total_samples * 100
            error_xgb = sum(comparison_df['XGBoost'] != comparison_df['Real']) / total_samples * 100

            # Exibir os percentuais de erro
            st.write(f"### Percentual de Erro")
            st.write(f"- Árvores de Decisão: **{error_tree:.2f}%**")
            st.write(f"- SVM: **{error_svm:.2f}%**")
            st.write(f"- Random Forest: **{error_rf:.2f}%**")
            st.write(f"- Logistic Regression: **{error_lr:.2f}%**")
            st.write(f"- KNN: **{error_knn:.2f}%**")
            st.write(f"- XGBoost: **{error_xgb:.2f}%**")

        # Comparativo de F1-Score entre os classificadores
        f1_scores = {
            'Árvores de Decisão': f1_tree,
            'SVM': f1_svm,
            'Random Forest': f1_rf,
            'Logistic Regression': f1_lr,
            'KNN': f1_knn,
            'XGBoost': f1_xgb
        }
        
        f1_score_df = pd.DataFrame(list(f1_scores.items()), columns=['Classificador', 'F1-Score'])

        fig_f1_score = px.bar(f1_score_df.sort_values(by='F1-Score', ascending=False), x='Classificador', y='F1-Score', color='Classificador',
                            title="Comparativo do F1-Score entre os Classificadores",
                            color_discrete_map=color_discrete_map_classificadores)
        
        st.plotly_chart(fig_f1_score)


        #Comparativo de Acurácia entre os classificadores
        accuracy_scores = {
            'Árvores de Decisão': accuracy_tree,
            'SVM': accuracy_svm,
            'Random Forest': accuracy_rf,
            'Logistic Regression': accuracy_lr,
            'KNN': accuracy_knn,
            'XGBoost': accuracy_xgb
        }

        accuracy_score_df = pd.DataFrame(list(accuracy_scores.items()), columns=['Classificador', 'Acurácia'])

        fig_accuracy_score = px.bar(accuracy_score_df.sort_values(by='Acurácia', ascending=False), x='Classificador', y='Acurácia', color='Classificador',
                                title="Comparativo da Acurácia entre os Classificadores",
                            color_discrete_map=color_discrete_map_classificadores)
        
        st.plotly_chart(fig_accuracy_score)

        #Comparativo de Precisão entre os classificadores
        precision_scores = {
            'Árvores de Decisão': precision_tree,
            'SVM': precision_svm,
            'Random Forest': precision_rf,
            'Logistic Regression': precision_lr,
            'KNN': precision_knn,
            'XGBoost': precision_xgb
        }

        precision_score_df = pd.DataFrame(list(precision_scores.items()), columns=['Classificador', 'Precisão'])

        fig_precision_score = px.bar(precision_score_df.sort_values(by='Precisão', ascending=False), x='Classificador', y='Precisão', color='Classificador',
                                title="Comparativo da Precisão entre os Classificadores",
                            color_discrete_map=color_discrete_map_classificadores)
        
        st.plotly_chart(fig_precision_score)

        #Comparativo de Recall entre os classificadores
        recall_scores = {
            'Árvores de Decisão': recall_tree,
            'SVM': recall_svm,
            'Random Forest': recall_rf,
            'Logistic Regression': recall_lr,
            'KNN': recall_knn,
            'XGBoost': recall_xgb
        }

        recall_score_df = pd.DataFrame(list(recall_scores.items()), columns=['Classificador', 'Recall'])

        fig_recall_score = px.bar(recall_score_df.sort_values(by='Recall', ascending=False), x='Classificador', y='Recall', color='Classificador',
                                title="Comparativo do Recall entre os Classificadores",
                            color_discrete_map=color_discrete_map_classificadores)
        
        st.plotly_chart(fig_recall_score)

        # Exibir um gráfico de barras polar para comparar as métricas de cada classificador
    

        fig_metrics_polar = go.Figure()
        
        # Adicionar as métricas de cada classificador ao gráfico polar
        fig_metrics_polar.add_trace(go.Scatterpolar(
            r=[accuracy_tree, precision_tree, recall_tree, f1_tree, error_tree/100],
            theta=['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'Erro'],
            fill=None,
            name='Árvores de Decisão',
            fillcolor='deepskyblue',
            opacity=0.70
        ))

        fig_metrics_polar.add_trace(go.Scatterpolar(
            r=[accuracy_svm, precision_svm, recall_svm, f1_svm, error_svm/100],
            theta=['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'Erro'],
            fill=None,
            name='SVM',
            fillcolor='maroon',
            opacity=0.70
        ))

        fig_metrics_polar.add_trace(go.Scatterpolar(
            r=[accuracy_rf, precision_rf, recall_rf, f1_rf, error_rf/100],
            theta=['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'Erro'],
            fill=None,
            name='Random Forest',
            fillcolor='darkblue',
            opacity=0.70
        ))

        fig_metrics_polar.add_trace(go.Scatterpolar(
            r=[accuracy_lr, precision_lr, recall_lr, f1_lr, error_lr/100],
            theta=['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'Erro'],
            fill=None,
            name='Logistic Regression',
            fillcolor='darkorange',
            opacity=0.60
        ))

        fig_metrics_polar.add_trace(go.Scatterpolar(
            r=[accuracy_knn, precision_knn, recall_knn, f1_knn, error_knn/100],
            theta=['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'Erro'],
            fill=None,
            name='KNN',
            fillcolor='darkviolet',
            opacity=0.70
        ))

        fig_metrics_polar.add_trace(go.Scatterpolar(
            r=[accuracy_xgb, precision_xgb, recall_xgb, f1_xgb, error_xgb/100],
            theta=['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'Erro'],
            fill=None,
            name='XGBoost',
            fillcolor='darkgreen',
            opacity=0.70
        ))

        fig_metrics_polar.update_layout(
            polar=dict(
                radialaxis=dict(
                    gridcolor='lightslategray',
                    tickcolor='lightslategray',
                    linecolor='#000000',
                    color='#000000',
                    visible=True,
                    range=[0, 1]
                ),
                ),
            showlegend=True,
            legend=dict(
                orientation='v',
                yanchor='top',
                y=0.9,
                xanchor='left',
                x=0.8,
            ),
            title="Comparativo de Métricas dos Classificadores"
        )

        st.plotly_chart(fig_metrics_polar)
    

    with st.expander("Playground", expanded=False):
        # Captura de valores numéricos para cada feature usando sliders
        input_data = {}
        # Supondo que você já tenha o scaler e a transformação logarítmica aplicados em X_train
        scaler = StandardScaler().fit(X_train)
        # Reverter os dados escalonados e a transformação logarítmica para mostrar os valores originais
        original_values = np.expm1(scaler.inverse_transform(X_train))

        with st.form("input_form"):
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["College", "English", "Quantitative", "Domain", "Analytical", "Study"])

            for i, feature in enumerate(X.columns):
                if feature != 'label':
                    min_val = float(original_values[:, i].min())
                    max_val = float(original_values[:, i].max())
                    if min_val >= max_val:
                        min_val = max_val - 1
                    mean_val = (min_val + max_val) / 2
                    if 'percentage' in feature:
                        input_data[feature] = tab1.slider(f"Valor para {feature}", min_value=min_val, max_value=max_val, value=min_val)
                    elif 'english' in feature:
                        input_data[feature] = tab2.slider(f"Valor para {feature}", min_value=min_val, max_value=max_val, value=min_val)
                    elif 'quantitative' in feature:
                        input_data[feature] = tab3.slider(f"Valor para {feature}", min_value=min_val, max_value=max_val, value=min_val)
                    elif 'domain' in feature:
                        input_data[feature] = tab4.slider(f"Valor para {feature}", min_value=min_val, max_value=max_val, value=min_val)
                    elif 'analytical' in feature:
                        input_data[feature] = tab5.slider(f"Valor para {feature}", min_value=min_val, max_value=max_val, value=min_val)
                    elif 'study' in feature:
                        input_data[feature] = tab6.slider(f"Valor para {feature}", min_value=min_val, max_value=max_val, value=min_val)
                    else:
                        input_data[feature] = st.slider(f"Valor para {feature}", min_value=min_val, max_value=max_val, value=min_val)

            submit_button = st.form_submit_button(label="Prever")

        # Prever a classe para a nova instância somente após a submissão do formulário
        if submit_button:
            # Criar um DataFrame com os valores originais selecionados
            new_instance_original = pd.DataFrame([input_data])

            # Aplicar a transformação logarítmica
            new_instance_log = np.log1p(new_instance_original)

            # Aplicar o StandardScaler
            new_instance_scaled = scaler.transform(new_instance_log)

            # Fazer a predição com os dados transformados para todos os classificadores
            try:
                pred_tree = clf_tree.predict(new_instance_scaled)[0]
                st.write(f'Árvores de Decisão prevê: {styled_metric_text_classes(class_mapping_binary[pred_tree], "")}', unsafe_allow_html=True)
            except Exception as e:
                st.write(f'Árvores de Decisão prevê: None', unsafe_allow_html=True)
                pass

            try:
                pred_svm = clf_svm.predict(new_instance_scaled)[0]
                st.write(f'SVM prevê: {styled_metric_text_classes(class_mapping_binary[pred_svm], "")}', unsafe_allow_html=True)
            except Exception as e:
                st.write(f'SVM prevê: None', unsafe_allow_html=True)
                pass

            try:
                pred_rf = clf_rf.predict(new_instance_scaled)[0]
                st.write(f'Random Forest prevê: {styled_metric_text_classes(class_mapping_binary[pred_rf], "")}', unsafe_allow_html=True)
            except Exception as e:
                st.write(f'Random Forest prevê: None', unsafe_allow_html=True)
                pass

            try:
                pred_lr = clf_lr.predict(new_instance_scaled)[0]
                st.write(f'Logistic Regression prevê: {styled_metric_text_classes(class_mapping_binary[pred_lr], "")}', unsafe_allow_html=True)
            except Exception as e:
                st.write(f'Logistic Regression prevê: None', unsafe_allow_html=True)
                pass

            try:
                pred_knn = clf_knn.predict(new_instance_scaled)[0]
                st.write(f'KNN prevê: {styled_metric_text_classes(class_mapping_binary[pred_knn], "")}', unsafe_allow_html=True)
            except Exception as e:
                st.write(f'KNN prevê: None', unsafe_allow_html=True)
                pass
            try:
                pred_xgb = clf_xgb.predict(new_instance_scaled)[0]
                st.write(f'XGBoost prevê: {styled_metric_text_classes(class_mapping_binary[pred_xgb], "")}', unsafe_allow_html=True)
            except Exception as e:
                st.write(f'XGBoost prevê: None', unsafe_allow_html=True)
                pass

if __name__ == '__main__':
    main()