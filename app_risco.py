import pandas as pd
import streamlit as st
import plotly.express as px

# função para carregar o dataset
@st.cache
def get_data():
    return pd.read_csv("risco.csv")


# função para treinar o modelo
def train_model():
    data = get_data()
    data = data.drop(columns='id_cliente')
    
    #Separando os Dados de Treino e de Teste

    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values


    #Redimensionando os Dados - Padronização com o StandardScaler
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    sc = StandardScaler()
    X_trans = sc.fit_transform(X)
    Maquina_preditiva = SVC(kernel='linear', gamma=0.1, C=10, random_state=7)
    Maquina_preditiva.fit(X_trans, y)
    return Maquina_preditiva

# criando um dataframe
data = get_data()

# treinando o modelo
model = train_model()

# título
st.title("Sistema de Previsão de Risco do Cliente para Concessão de Empréstimos")

# subtítulo
st.markdown("Este é um Aplicativo utilizado para exibir a solução de Ciência de Dados para o problema de predição de Risco do Cliente para concessão de empréstimos.")

st.sidebar.subheader("Insira os Dados dos Clientes para Análise do Risco")

# mapeando dados do usuário para cada atributo
indice_inad = st.sidebar.number_input("Índice de Inadimplência", value=data.indice_inad.mean())
anot_cadastrais = st.sidebar.number_input("Anotações Cadastrais", value=data.anot_cadastrais.mean())
class_renda = st.sidebar.number_input("Classificação da Renda", value=data.class_renda.mean())
saldo_contas = st.sidebar.number_input("Saldo de Contas", value=data.saldo_contas.mean())

# inserindo um botão na tela
btn_predict = st.sidebar.button("Realizar Predição do Risco")

# verifica se o botão foi acionado
if btn_predict:
    result = model.predict([[indice_inad,anot_cadastrais,class_renda,saldo_contas]])
    st.subheader("O Risco Previsto do Cliente é:")
    result = result[0]
    st.write(result)

# verificando o dataset
st.subheader("Selecionando as Variáveis de análise dos clientes")

# atributos para serem exibidos por padrão
defaultcols = ["anot_cadastrais","indice_inad","class_renda","saldo_contas"]

# defindo atributos a partir do multiselect
cols = st.multiselect("Atributos", data.columns.tolist(), default=defaultcols)

# exibindo os top 8 registro do dataframe
st.dataframe(data[cols].head(8))