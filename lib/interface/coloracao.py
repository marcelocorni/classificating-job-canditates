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

# Aplicar a formatação condicional
def apply_styles(x):
    # Inicializar uma lista para armazenar os estilos
    styles = ['' for _ in range(len(x))]

    # Iterar sobre as colunas
    for i, col in enumerate(x.index):
        if col == 'Real':
            styles[i] = 'color: blue'  # Coluna 'Real' em azul
        elif x[col] == x['Real']:
            styles[i] = 'color: green'  # Valores corretos em verde
        else:
            styles[i] = 'color: red'  # Valores errados em vermelho
    return styles