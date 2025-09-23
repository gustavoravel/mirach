from django.urls import path
from . import views

app_name = 'datasets'

urlpatterns = [
    path('', views.dataset_list, name='list'),
    path('upload/', views.dataset_upload, name='upload'),
    path('<int:pk>/', views.dataset_detail, name='detail'),
    path('<int:pk>/preview/', views.dataset_preview, name='preview'),
    path('<int:pk>/explore/', views.dataset_explore, name='explore'),
    path('<int:pk>/edit/', views.dataset_edit, name='edit'),
    path('<int:pk>/delete/', views.dataset_delete, name='delete'),
    path('<int:pk>/mapping/', views.column_mapping, name='mapping'),
    path('<int:pk>/process/', views.dataset_process, name='process'),
    path('<int:pk>/backtest/', views.dataset_backtest, name='backtest'),
    # API
    path('api/v1/datasets/', views.api_list_datasets, name='api_list_datasets'),
]
