import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------------------------
# 1. CARREGAMENTO DOS DADOS
# -----------------------------------------------------------------------------
df = pd.read_csv("pizzas.csv")
df.columns = df.columns.str.strip()

# -----------------------------------------------------------------------------
# 2. TREINAMENTO DA INTELIGÊNCIA ARTIFICIAL
# -----------------------------------------------------------------------------
X = df[['diametro']]
y = df['preco']

modelo = LinearRegression()
modelo.fit(X, y)

# -----------------------------------------------------------------------------
# 3. INTERFACE VISUAL DO SITE (Streamlit)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Preditor de Preço de Pizza", page_icon="🍕")

st.title("Previsão de Preço de Pizzas 🍕")
st.write("Insira o diâmetro desejado para que a Inteligência Artificial calcule o valor estimado.")

# Criando o campo numérico interativo na tela
tamanho_usuario = st.number_input(
    label="Diâmetro da pizza (em centímetros):", 
    min_value=10.0, 
    max_value=1000.0, 
    value=32.0, 
    step=1.0
)

# Botão de cálculo e exibição do resultado
if st.button("Calcular Preço Estimado"):
    dados_teste = pd.DataFrame({'diametro': [tamanho_usuario]})
    preco_previsto = modelo.predict(dados_teste)
    
    # CORREÇÃO AQUI: Adicionado [0] para pegar o número puro dentro da array
    st.success(f"O preço estimado para uma pizza de {tamanho_usuario}cm é: R$ {preco_previsto[0]:.2f}")

# -----------------------------------------------------------------------------
# 4. GRÁFICO INTERATIVO
# -----------------------------------------------------------------------------
st.subheader("Visualização dos Dados do CSV 📊")
st.write("Veja abaixo a distribuição dos preços atuais que a IA usou para aprender:")

# CORREÇÃO AQUI: Mudado de use_container_width=True para width='stretch'
st.scatter_chart(
    data=df,
    x="diametro",
    y="preco",
    color="#FF4B4B",
    width="stretch"
)
