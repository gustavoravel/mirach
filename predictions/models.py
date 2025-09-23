from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone


class PredictionModel(models.Model):
    """Modelo para algoritmos de previsão disponíveis"""
    
    ALGORITHM_CHOICES = [
        ('arima', 'ARIMA'),
        ('ets', 'ETS (Exponential Smoothing)'),
        ('prophet', 'Prophet'),
        ('lstm', 'LSTM'),
        ('linear_regression', 'Regressão Linear'),
        ('polynomial_regression', 'Regressão Polinomial'),
        ('ridge_regression', 'Regressão Ridge'),
        ('lasso_regression', 'Regressão Lasso'),
        ('random_forest', 'Random Forest'),
        ('xgboost', 'XGBoost'),
        ('svr', 'Support Vector Regression'),
        ('neural_network', 'Rede Neural'),
    ]
    
    name = models.CharField(max_length=100, verbose_name="Nome do Algoritmo")
    algorithm_type = models.CharField(max_length=50, choices=ALGORITHM_CHOICES, 
                                     verbose_name="Tipo do Algoritmo")
    description = models.TextField(verbose_name="Descrição")
    parameters = models.JSONField(default=dict, blank=True, verbose_name="Parâmetros")
    is_active = models.BooleanField(default=True, verbose_name="Ativo")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Criado em")
    
    class Meta:
        verbose_name = "Modelo de Previsão"
        verbose_name_plural = "Modelos de Previsão"
        ordering = ['name']
    
    def __str__(self):
        return self.name


class Prediction(models.Model):
    """Modelo para previsões realizadas"""
    
    STATUS_CHOICES = [
        ('pending', 'Pendente'),
        ('training', 'Treinando'),
        ('completed', 'Concluído'),
        ('failed', 'Falhou'),
    ]
    
    name = models.CharField(max_length=200, verbose_name="Nome da Previsão")
    description = models.TextField(blank=True, verbose_name="Descrição")
    project = models.ForeignKey('projects.Project', on_delete=models.CASCADE, 
                               related_name='predictions', verbose_name="Projeto")
    dataset = models.ForeignKey('datasets.Dataset', on_delete=models.CASCADE, 
                               verbose_name="Dataset")
    prediction_model = models.ForeignKey(PredictionModel, on_delete=models.CASCADE, 
                                        verbose_name="Modelo de Previsão")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, 
                             default='pending', verbose_name="Status")
    
    # Parâmetros da previsão
    prediction_horizon = models.IntegerField(verbose_name="Horizonte de Previsão")
    train_size = models.FloatField(default=0.8, verbose_name="Tamanho do Treino")
    validation_size = models.FloatField(default=0.1, verbose_name="Tamanho da Validação")
    test_size = models.FloatField(default=0.1, verbose_name="Tamanho do Teste")
    
    # Resultados
    model_parameters = models.JSONField(default=dict, blank=True, verbose_name="Parâmetros do Modelo")
    metrics = models.JSONField(default=dict, blank=True, verbose_name="Métricas")
    predictions_data = models.JSONField(default=list, blank=True, verbose_name="Dados da Previsão")
    
    # Metadados
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="Criado por")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Criado em")
    completed_at = models.DateTimeField(null=True, blank=True, verbose_name="Concluído em")
    error_message = models.TextField(blank=True, verbose_name="Mensagem de Erro")
    
    class Meta:
        verbose_name = "Previsão"
        verbose_name_plural = "Previsões"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.prediction_model.name})"
    
    @property
    def is_completed(self):
        return self.status == 'completed'
    
    @property
    def is_failed(self):
        return self.status == 'failed'


class PredictionResult(models.Model):
    """Modelo para armazenar resultados detalhados das previsões"""
    
    prediction = models.ForeignKey(Prediction, on_delete=models.CASCADE, 
                                  related_name='results', verbose_name="Previsão")
    timestamp = models.DateTimeField(verbose_name="Timestamp")
    actual_value = models.FloatField(null=True, blank=True, verbose_name="Valor Real")
    predicted_value = models.FloatField(verbose_name="Valor Previsto")
    confidence_interval_lower = models.FloatField(null=True, blank=True, 
                                                 verbose_name="Intervalo de Confiança Inferior")
    confidence_interval_upper = models.FloatField(null=True, blank=True, 
                                                 verbose_name="Intervalo de Confiança Superior")
    
    class Meta:
        verbose_name = "Resultado da Previsão"
        verbose_name_plural = "Resultados das Previsões"
        ordering = ['timestamp']
    
    def __str__(self):
        return f"{self.prediction.name} - {self.timestamp}"