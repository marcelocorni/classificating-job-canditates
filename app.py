import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
from sklearn.decomposition import PCA
import hashlib

st.set_page_config(page_title='Trabalho 5 - Clasificação', layout='wide')
st.set_option('deprecation.showPyplotGlobalUse', False)
pl.style.use('dark_background')

st.title("Classificação dos Dados Agrupados de Candidatos a Emprego")

# Função para gerar o URL do Gravatar a partir do e-mail
def get_gravatar_url(email, size=100):
    email_hash = hashlib.md5(email.strip().lower().encode('utf-8')).hexdigest()
    gravatar_url = f"https://www.gravatar.com/avatar/{email_hash}?s={size}"
    return gravatar_url

# Definir o e-mail e o tamanho da imagem
email = "marcelo@desenvolvedor.net"  # Substitua pelo seu e-mail
size = 200  # Tamanho da imagem

# Obter o URL do Gravatar
gravatar_url = get_gravatar_url(email, size)

# Layout principal com colunas
col1, col2 = st.columns([1, 2])

# Conteúdo da coluna esquerda
with col1:
    st.markdown(
        f"""
        <div style="text-align: right;">
            <img src="{gravatar_url}" alt="Gravatar" style="width: 150px;">
        </div>
        """,
        unsafe_allow_html=True
    )
# Conteúdo da coluna direita
with col2:
    st.write("### Análise Classificativa de Dados Agrupados")
    st.write("###### Marcelo Corni Alves")
    st.write("Agosto/2024")
    st.write("Disciplina: Mineração de Dados")

# Carregar os dados
data = pd.read_csv('data/dados_normalizados.csv')

# Adicionar nova coluna label para classificação BP ou NAO_BP
data['label'] = data['performance'].apply(lambda x: 1 if x == 'BP' else 0)

# Remover a coluna 'performance' do DataFrame para não ser incluída no playground
X = data.drop(columns=['performance', 'label'])
y = data['label']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parâmetros para Árvores de Decisão
with st.sidebar.expander("Hiperparâmetros - Árvores de Decisão",expanded=False):
    max_depth_tree = st.slider("Profundidade Máxima", 1, 20, 5, key="max_depth_tree")
    min_samples_split_tree = st.slider("Mínimo de Amostras para Split", 2, 20, 2, key="min_samples_split_tree")
    min_samples_leaf_tree = st.slider("Mínimo de Amostras na Folha", 1, 10, 1, key="min_samples_leaf_tree")
    max_features_tree = st.selectbox("Máximo de Features", ['sqrt', 'log2', None], key="max_features_tree")
    criterion_tree = st.selectbox("Critério", ["gini", "entropy","log_loss"], key="criterion_tree")

clf_tree = DecisionTreeClassifier(max_depth=max_depth_tree,
                                  min_samples_split=min_samples_split_tree,
                                  min_samples_leaf=min_samples_leaf_tree,
                                  max_features=max_features_tree,
                                  criterion=criterion_tree,
                                  random_state=42)
clf_tree.fit(X_train, y_train)
y_pred_tree = clf_tree.predict(X_test)

# Parâmetros para SVM
with st.sidebar.expander("Hiperparâmetros - SVM",expanded=False):
    C_svm = st.slider("C (Regularização)", 0.01, 10.0, 1.0, key="C_svm")
    kernel_svm = st.selectbox("Kernel", ['linear', 'poly', 'rbf', 'sigmoid'], key="kernel_svm")
    degree_svm = st.slider("Grau (se kernel for poly)", 2, 5, 3, key="degree_svm")
    gamma_svm = st.selectbox("Gamma", ['scale', 'auto'], key="gamma_svm")
    coef0_svm = st.slider("Coef0 (para poly/sigmoid)", 0.0, 1.0, 0.0, key="coef0_svm")

clf_svm = SVC(C=C_svm, kernel=kernel_svm, degree=degree_svm, gamma=gamma_svm, coef0=coef0_svm, random_state=42)
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)

# Parâmetros para Random Forest
with st.sidebar.expander("Hiperparâmetros - Random Forest",expanded=False):
    n_estimators_rf = st.slider("Número de Árvores (Estimadores)", 10, 200, 100, key="n_estimators_rf")
    max_depth_rf = st.slider("Profundidade Máxima", 1, 20, 10, key="max_depth_rf")
    min_samples_split_rf = st.slider("Mínimo de Amostras para Split", 2, 20, 2, key="min_samples_split_rf")
    min_samples_leaf_rf = st.slider("Mínimo de Amostras na Folha", 1, 10, 1, key="min_samples_leaf_rf")
    max_features_rf = st.selectbox("Máximo de Features", ['sqrt', 'log2', None], key="max_features_rf")
    criterion_rf = st.selectbox("Critério", ["gini", "entropy","log_loss"], key="criterion_rf")
    bootstrap_rf = st.selectbox("Bootstrap", [True, False], key="bootstrap_rf")

clf_rf = RandomForestClassifier(n_estimators=n_estimators_rf,
                                max_depth=max_depth_rf,
                                min_samples_split=min_samples_split_rf,
                                min_samples_leaf=min_samples_leaf_rf,
                                max_features=max_features_rf,
                                bootstrap=bootstrap_rf,
                                criterion=criterion_rf,
                                random_state=42)

clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

# Converter as predições para as classes nomeadas
class_mapping_binary = {0: "NAO_BP", 1: "BP"}
y_pred_rf_mapped = [class_mapping_binary.get(pred, "Desconhecido") for pred in y_pred_rf]


# Função para aplicar cores com base nos valores
def color_metric(val):
    if val > 0.75:
        color = 'green'
    elif val > 0.6:
        color = 'orange'
    else:
        color = 'red'
    return f'color: {color}'

# Funções para estilizar as métricas
def styled_metric_text(metric_value, metric_name):
    color = color_metric(metric_value)[7:]  # Remove 'color: ' para pegar só a cor
    return f'<span style="color:{color}; font-weight:bold;">{metric_name} {metric_value:.2f}</span>'

# Função para aplicar cores com base nos classes
def color_metric_classes(str):
    if str == 'BP':
        color = 'green'
    elif str == 'MP':
        color = 'orange'
    elif str == 'LP':
        color = 'red'
    else:
        color = 'purple'
    return f'color: {color}'

# Funções para estilizar as classes
def styled_metric_text_classes(metric_value, metric_name):
    color = color_metric_classes(metric_value)[7:]  # Remove 'color: ' para pegar só a cor
    return f'<span style="color:{color}; font-weight:bold;">{metric_name} {metric_value}</span>'


# Função para criar dataframe do classification_report
def classification_report_to_df(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:  # Modificado para evitar linhas inválidas
        row_data = line.split()
        if len(row_data) == 0 or len(row_data) < 5:  # Ignorar linhas inválidas
            continue
        row = {
            'class': row_data[0],
            'precision': float(row_data[1]),
            'recall': float(row_data[2]),
            'f1-score': float(row_data[3]),
            'support': int(row_data[4])
        }
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    return dataframe.set_index('class')

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


with st.expander("Comparação dos Classificadores", expanded=False):
    # Comparação da distribuição das classes preditas pelos três classificadores

    fig_comparison = px.histogram(comparison_data, barmode='group', color_discrete_map=color_discrete_map_binary,
                                labels={'value': 'Classe Predita', 'variable': 'Classificador'},
                                title="Distribuição das Classes Preditas")
    st.plotly_chart(fig_comparison)


    # Tabela de comparação das previsões
    comparison_df = pd.DataFrame({
        'Real': [class_mapping_binary.get(real, "Desconhecido") for real in y_test],
        'Árvores de Decisão': [class_mapping_binary.get(pred, "Desconhecido") for pred in y_pred_tree],
        'SVM': [class_mapping_binary.get(pred, "Desconhecido") for pred in y_pred_svm],
        'Random Forest': [class_mapping_binary.get(pred, "Desconhecido") for pred in y_pred_rf]
    })

    # Aplicar a formatação condicional
    comparison_styled = comparison_df.style.apply(lambda x: ['background-color: red' if x[col] != x['Real'] else '' 
                                                            for col in x.index], axis=1)
    
    st.write("### Tabela de Comparação das Previsões")
    st.dataframe(comparison_styled)

    # Cálculo do percentual de erro para cada classificador
    total_samples = len(y_test)
    error_tree = sum(comparison_df['Árvores de Decisão'] != comparison_df['Real']) / total_samples * 100
    error_svm = sum(comparison_df['SVM'] != comparison_df['Real']) / total_samples * 100
    error_rf = sum(comparison_df['Random Forest'] != comparison_df['Real']) / total_samples * 100

    # Exibir os percentuais de erro
    st.write(f"### Percentual de Erro")
    st.write(f"- Árvores de Decisão: **{error_tree:.2f}%**")
    st.write(f"- SVM: **{error_svm:.2f}%**")
    st.write(f"- Random Forest: **{error_rf:.2f}%**")

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
                mean_val = float(original_values[:, i].mean())
                if 'percentage' in feature:
                    input_data[feature] = tab1.slider(f"Valor para {feature}", min_value=min_val, max_value=max_val, value=mean_val)
                elif 'english' in feature:
                    input_data[feature] = tab2.slider(f"Valor para {feature}", min_value=min_val, max_value=max_val, value=mean_val)
                elif 'quantitative' in feature:
                    input_data[feature] = tab3.slider(f"Valor para {feature}", min_value=min_val, max_value=max_val, value=mean_val)
                elif 'domain' in feature:
                    input_data[feature] = tab4.slider(f"Valor para {feature}", min_value=min_val, max_value=max_val, value=mean_val)
                elif 'analytical' in feature:
                    input_data[feature] = tab5.slider(f"Valor para {feature}", min_value=min_val, max_value=max_val, value=mean_val)
                elif 'study' in feature:
                    input_data[feature] = tab6.slider(f"Valor para {feature}", min_value=min_val, max_value=max_val, value=mean_val)
                else:
                    input_data[feature] = st.slider(f"Valor para {feature}", min_value=min_val, max_value=max_val, value=mean_val)

        submit_button = st.form_submit_button(label="Prever")

    # Prever a classe para a nova instância somente após a submissão do formulário
    if submit_button:
        # Criar um DataFrame com os valores originais selecionados
        new_instance_original = pd.DataFrame([input_data])

        # Aplicar a transformação logarítmica
        new_instance_log = np.log1p(new_instance_original)

        # Aplicar o StandardScaler
        new_instance_scaled = scaler.transform(new_instance_log)

        # Fazer a predição com os dados transformados
        pred_tree = clf_tree.predict(new_instance_scaled)[0]
        pred_svm = clf_svm.predict(new_instance_scaled)[0]
        pred_rf = clf_rf.predict(new_instance_scaled)[0]

        st.write(f'Árvores de Decisão prevê: {styled_metric_text_classes(class_mapping_binary[pred_tree], "")}', unsafe_allow_html=True)
        st.write(f'SVM prevê: {styled_metric_text_classes(class_mapping_binary[pred_svm], "")}', unsafe_allow_html=True)
        st.write(f'Random Forest prevê: {styled_metric_text_classes(class_mapping_binary[pred_rf], "")}', unsafe_allow_html=True)



with st.expander('Conclusão', expanded=False):
    st.write("""
    A análise dos resultados dos três classificadores — Árvores de Decisão, SVM e Random Forest — revela comportamentos distintos em relação à predição das classes, destacando as forças e limitações de cada método.

    - **Árvores de Decisão**: Este classificador mostrou-se eficaz em termos de interpretabilidade, permitindo compreender claramente o processo de decisão. No entanto, observamos que ele tende a apresentar dificuldades em capturar padrões mais complexos nos dados, o que pode ter levado a um percentual de erro mais elevado em certas classes. Esse comportamento reflete uma possível tendência ao overfitting em situações onde os dados não seguem regras simples e bem definidas.

    - **SVM (Support Vector Machine)**: O SVM destacou-se por sua capacidade de lidar com margens de decisão bem definidas, o que resultou em um desempenho robusto nas classes analisadas. No entanto, a análise dos resultados mostra que a sensibilidade do SVM à escolha dos hiperparâmetros pode impactar significativamente o desempenho, tornando o processo de ajuste fino crucial para a obtenção de bons resultados. Embora menos interpretável que as Árvores de Decisão, o SVM conseguiu capturar padrões complexos de forma eficaz, resultando em um menor percentual de erro em comparação com as Árvores de Decisão.

    - **Random Forest**: Este classificador apresentou um equilíbrio notável entre acurácia e interpretabilidade. A utilização de múltiplas árvores (ensembles) ajudou a melhorar a precisão das predições e a reduzir a variabilidade do modelo. Em nossas análises, o Random Forest se mostrou particularmente robusto, mantendo um percentual de erro competitivo e, em muitos casos, inferior ao dos outros classificadores. Sua capacidade de evitar o overfitting ao combinar os resultados de várias árvores o torna uma escolha poderosa, especialmente em contextos onde a precisão é fundamental.

    ### Desempenho Geral
    Cada técnica demonstrou pontos fortes específicos, que se refletem nos percentuais de erro observados. O SVM e o Random Forest, em particular, mostraram-se mais eficazes em capturar a complexidade dos dados, resultando em uma melhor performance geral. A escolha entre esses classificadores deve ser guiada pelo contexto específico da aplicação, considerando a necessidade de interpretabilidade, a complexidade dos dados e os requisitos de precisão.
    """)

