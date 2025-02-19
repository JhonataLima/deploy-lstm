from sklearn.preprocessing import MinMaxScaler
import pickle


def normalizador(df):
    # Supondo que você já tenha treinado o normalizador
    normalizer = MinMaxScaler(feature_range=(0, 1))
    normalizer.fit((df))