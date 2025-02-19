from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn
from sklearn.preprocessing import MinMaxScaler


app = FastAPI()

# Carregando o modelo
try:
    with open('model/lstm.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Erro ao carregar o modelo: {str(e)}")

class PriceData(BaseModel):
    historical_prices: list

@app.post("/predict")
def predict_price(data: PriceData):
    try:
        # Converte dados de entrada em array numpy
        input_data = np.array(data.historical_prices).reshape(1, -1)
        # normalize = MinMaxScaler(feature_range=(0, 1))
        
        # Fazer previsão
        prediction = model.predict(input_data)
        
        # Retorna a previsão
        return {"predicted_price": prediction[0].tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao fazer previsão: {str(e)}")

# Executa aplicação
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import pickle
# import numpy as np
# import uvicorn
# from sklearn.preprocessing import MinMaxScaler

# app = FastAPI()

# # Carregando o modelo
# try:
#     with open('model/lstm.pkl', 'rb') as file:
#         model = pickle.load(file)
# except Exception as e:
#     raise HTTPException(status_code=500, detail=f"Erro ao carregar o modelo: {str(e)}")

# # Carregando o normalizador
# try:
#     with open('model/normalizer.pkl', 'rb') as file:
#         normalizer = pickle.load(file)
# except Exception as e:
#     raise HTTPException(status_code=500, detail=f"Erro ao carregar o normalizador: {str(e)}")

# class PriceData(BaseModel):
#     historical_prices: list

# @app.post("/predict")
# def predict_price(data: PriceData):
#     try:

#         # Converte dados de entrada em array numpy
#         input_data = np.array(data.historical_prices).reshape(-1, 1)
        
#         # Normalizar os dados de entrada
#         input_data_normalized = normalizer.transform(input_data).reshape(1, -1)
        
#         # Fazer previsão
#         prediction = model.predict(input_data_normalized)
        
#         # Desnormalizar a previsão
#         prediction_desnormalized = normalizer.inverse_transform(prediction.reshape(-1, 1))
        
#         # Retorna a previsão desnormalizada
#         return {"predicted_price": prediction_desnormalized[0][0].tolist()}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Erro ao fazer previsão: {str(e)}")

# # Executa aplicação
# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)