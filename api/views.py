from rest_framework import viewsets
from rest_framework.response import Response
from .models import ARIMAModel, ETSModel
from .prediction import ARIMA_Model, ETS_Model, preprocess_data

class ARIMATrainViewSet(viewsets.ViewSet):
    def create(self, request):
        # Obtenha os dados da solicitação
        data = request.data.get('data')
        column_pair = request.data.get('column_pair')

        # Pré-processar os dados
        dataset = preprocess_data(data, column_pair)

        # Treinar o modelo ARIMA
        arima_model = ARIMA_Model(dataset, column_pair)
        arima_model.fit()
        arima_model.save_model()

        return Response({'message': 'Modelo ARIMA treinado com sucesso'})

class ETSTrainViewSet(viewsets.ViewSet):
    def create(self, request):
        # Obtenha os dados da solicitação
        data = request.data.get('data')
        column_pair = request.data.get('column_pair')

        # Pré-processar os dados
        dataset = preprocess_data(data, column_pair)

        # Treinar o modelo ETS
        ets_model = ETS_Model(dataset, column_pair)
        ets_model.fit()
        ets_model.save_model()

        return Response({'message': 'Modelo ETS treinado com sucesso'})

class PredictionViewSet(viewsets.ViewSet):
    def create(self, request):
        # Obtenha os dados da solicitação
        new_data = request.data.get('new_data')

        # Faça previsões com base nos modelos treinados (você precisa implementar a lógica aqui)
        # Exemplo:
        arima_predictions = arima_model.predict(new_data)
        ets_predictions = ets_model.predict(new_data)

        return Response({'arima_predictions': arima_predictions, 'ets_predictions': ets_predictions})
