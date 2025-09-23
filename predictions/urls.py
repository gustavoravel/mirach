from django.urls import path
from . import views

app_name = 'predictions'

urlpatterns = [
    path('', views.prediction_list, name='list'),
    path('create/', views.prediction_create, name='create'),
    path('wizard/', views.prediction_wizard, name='wizard'),
    path('<int:pk>/', views.prediction_detail, name='detail'),
    path('<int:pk>/run/', views.prediction_run, name='run'),
    path('<int:pk>/results/', views.prediction_results, name='results'),
    path('<int:pk>/delete/', views.prediction_delete, name='delete'),
    path('models/', views.prediction_models, name='models'),
    
    # API endpoints
    path('api/recommendations/<int:dataset_id>/', views.get_model_recommendations, name='recommendations'),
    path('api/compare/<int:dataset_id>/', views.compare_models, name='compare'),
    path('api/backtest/<int:dataset_id>/', views.backtest, name='backtest'),
    path('api/visualization/<int:pk>/', views.get_visualization_data, name='visualization'),
    path('api/status/<int:pk>/', views.prediction_status, name='status'),
    path('api/export/<int:pk>/csv/', views.export_results_csv, name='export_csv'),
    path('api/export/<int:pk>/json/', views.export_results_json, name='export_json'),
]
