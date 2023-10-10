from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ARIMATrainViewSet, ETSTrainViewSet, PredictionViewSet

router = DefaultRouter()
router.register(r'arima-train', ARIMATrainViewSet, basename='arima-train')
router.register(r'ets-train', ETSTrainViewSet, basename='ets-train')
router.register(r'predict', PredictionViewSet, basename='predict')

urlpatterns = [
    path('', include(router.urls)),
]
