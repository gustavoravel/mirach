import pickle
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import arma_order_select_ic
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# pareamento das colunas
# exemplo:
"""column_pair = {
    'item_id': 'Dept',
    'timestamp': 'Date',
    'demand': 'Weekly_Sales'
}"""


class ARIMA_Model:
    def __init__(self, dataset, column_pair):
        self.dataset = dataset
        self.column_pair = column_pair
        self.best_model = None
        self.best_order = None

    def fit(self):
        data = self.dataset[self.column_pair['demand']]
        # selecionando as ordens 'p' e 'q' com base no critério AIC e BIC
        order_aic = arma_order_select_ic(data, ic='aic')
        order_bic = arma_order_select_ic(data, ic='bic')
        # escolhendo a ordem com o menor AIC e BIC
        if order_aic.aic_min_order[2] < order_bic.bic_min_order[2]:
            best_order = order_aic.aic_min_order[:2]
        else:
            best_order = order_bic.bic_min_order[:2]
        # treinando o modelo com as ordens selecionadas
        model = ARIMA(data, order=best_order)
        model_fit = model.fit()
        self.best_model = model_fit
        self.best_order = best_order
        return model_fit

    def save_model(self):
        # salvando o modelo em um arquivo .pkl
        pickle.dump(self.best_model, open('best_model.pkl', 'wb'))

    def evaluate(self):
        # avaliando a precisão do modelo
        print(f'Ordens selecionadas: {self.best_order}')
        print(f'AIC: {self.best_model.aic}')
        print(f'BIC: {self.best_model.bic}')

    def predict(self, new_data):
        # carregando o modelo salvo
        best_model = pickle.load(open('best_model.pkl', 'rb'))
        # fazendo as previsões com os novos dados
        predictions, _, _ = best_model.forecast(steps=len(new_data), exog=new_data[self.column_pair['item_id']])
        return predictions


class ETS_Model:
    def __init__(self, dataset, column_pair):
        self.dataset = dataset
        self.column_pair = column_pair
        self.best_model = None
        self.best_params = None

    def fit(self):
        data = self.dataset[self.column_pair['demand']]
        # treinando o modelo com diferentes parâmetros
        best_aic = float('inf')
        for trend in ['add', 'mul', None]:
            for damped in [True, False]:
                for seasonal in ['add', 'mul', None]:
                    model = ExponentialSmoothing(data, trend=trend, damped=damped, seasonal=seasonal)
                    model_fit = model.fit()
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        self.best_model = model_fit
                        self.best_params = {'trend': trend, 'damped': damped, 'seasonal': seasonal}
        return self.best_model

    def save_model(self):
        # salvando o modelo em um arquivo .pkl
        pickle.dump(self.best_model, open('best_model.pkl', 'wb'))

    def evaluate(self):
        # avaliando a precisão do modelo
        print(f'Parâmetros selecionados: {self.best_params}')
        print(f'AIC: {self.best_model.aic}')

    def predict(self, new_data):
        # carregando o modelo salvo
        model = pickle.load(open('best_model.pkl', 'rb'))
        # fazendo novas previsões
        predictions = model.predict(new_data)
        return predictions


def predict(data, prediction_variables, chosen_algorithm, period_number, period_type, prediction_frequency):
    pass
