# Bibliotecas básicas
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Decomposição sazonal
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

# LSTM
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.metrics import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler

# Função para baixar dados históricos do Yahoo Finance
def download_data(ticker="^BVSP", period="5y", interval="1d"):
    """
    Baixa os dados históricos de um ativo no Yahoo Finance.
    """
    try:
        df = yf.download(ticker, period=period, interval=interval)
        df['ds'] = df.index
        df = df[['ds', 'Close']]
        df.rename(columns={'Close': 'y'}, inplace=True)
        return df
    except Exception as e:
        print(f"Erro ao baixar os dados: {e}")

# Função para preparar os dados de treino e teste
def prepare_data(df, previsao_dias):
    """
    Prepara os dados de treino e teste para o modelo LSTM.
    """
    treino = df[:date.today() - timedelta(days=1)]
    teste = df[date.today():]
    
    normalize = MinMaxScaler(feature_range=(0, 1))
    df_treino_mms = normalize.fit_transform(treino.iloc[:, 1:2].values)
    
    x_treino = []
    y_treino = []
    
    # Construindo os dados de treino
    for i in range(previsao_dias, len(df_treino_mms)):
        x_treino.append(df_treino_mms[i-previsao_dias:i, 0])
        y_treino.append(df_treino_mms[i, 0])
    
    x_treino, y_treino = np.array(x_treino), np.array(y_treino)
    x_treino = np.reshape(x_treino, (x_treino.shape[0], x_treino.shape[1], 1))
    
    return treino, teste, df_treino_mms, x_treino, y_treino, normalize

# Função para construir o modelo LSTM
def build_lstm_model(input_shape, unit, drop):
    """
    Constrói e compila o modelo LSTM.
    """
    modelo = Sequential()
    modelo.add(LSTM(units=unit, return_sequences=True, input_shape=input_shape))
    modelo.add(Dropout(drop))
    modelo.add(LSTM(units=unit, return_sequences=True))
    modelo.add(Dropout(drop))
    modelo.add(LSTM(units=unit, return_sequences=True))
    modelo.add(Dropout(drop))
    modelo.add(LSTM(units=unit))
    modelo.add(Dropout(drop))
    modelo.add(Dense(units=1, activation='linear'))
    modelo.compile(optimizer='adam', loss='mean_squared_error', metrics=[MeanSquaredError()])
    # modelo.compile(optimizer='adam', loss='mean_squared_error', metrics=[keras.metrics.MeanSquaredError])
    return modelo

# Função para treinar o modelo LSTM
def train_lstm_model(modelo, x_treino, y_treino, epoch, batch_size=32):
    """
    Treina o modelo LSTM.
    """
    modelo.fit(x_treino, y_treino, epochs=epoch, batch_size=batch_size)
    return modelo

# Função para preparar os dados de teste
def prepare_test_data(df_completo, treino, teste, previsao_dias, normalize):
    """
    Prepara os dados para testar o modelo LSTM.
    """
    df_completo = pd.concat((treino['y'], teste['y']), axis=0)
    modelo_entrada = df_completo[(len(df_completo) - len(teste) - previsao_dias):].values
    modelo_entrada = modelo_entrada.reshape(-1, 1)
    modelo_entrada = normalize.transform(modelo_entrada)
    
    x_teste2 = []
    for i in range(previsao_dias, len(modelo_entrada)):
        x_teste2.append(modelo_entrada[i-previsao_dias:i, 0])
    
    x_teste2 = np.array(x_teste2)
    x_teste2 = np.reshape(x_teste2, (x_teste2.shape[0], x_teste2.shape[1], 1))
    return x_teste2

# Função para calcular o MAPE (Mean Absolute Percentage Error)
def calculate_mape(y_real, y_pred):
    """
    Calcula o MAPE (Mean Absolute Percentage Error).
    """
    mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
    return mape

# Função para gerar previsões
def generate_forecast(modelo, x_teste2, normalize, y_real=None):
    """
    Gera previsões a partir do modelo LSTM.
    """
    previsoes = modelo.predict(x_teste2)
    previsoes = normalize.inverse_transform(previsoes)
    
    # Calcular o MAPE, se os valores reais forem fornecidos
    if y_real is not None:
        mape = calculate_mape(y_real, previsoes)
        print(f"MAPE: {mape:.2f}%")  # Exibir o MAPE no console
    
    return previsoes

# Função para gerar o gráfico com dados históricos e previsões
def generate_plot(df, previsoes, start_date='2024-12-10', forecast_start='2024-12-17'):
    """
    Gera um gráfico com os dados históricos e as previsões.
    """
    fig = make_subplots(rows=1, cols=1)
    
    # Dados históricos
    fig.append_trace(go.Scatter(x=df[start_date:].index.values, y=df[start_date:].y.values.flatten()), row=1, col=1)
    
    # Previsões
    fig.append_trace(go.Scatter(x=df[forecast_start:].index, y=pd.DataFrame(previsoes)[0]), row=1, col=1)
    
    fig.update_layout(
        height=700, 
        width=1500, 
        title_text='Prevendo Valor', 
        showlegend=False, 
        template='plotly_dark'
    )
    
    fig.show()
