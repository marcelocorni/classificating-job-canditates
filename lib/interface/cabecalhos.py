def cria_cabecalho_padrao(st, titulo, subtitulo, disciplina ,data):
    
    import hashlib
    
    # Título da aplicação
    st.write(f'### {titulo}')

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
        st.write(f"### {subtitulo}")
        st.write("###### Marcelo Corni Alves")
        st.write(f"{data}")
        st.write(f"Disciplina: {disciplina}")