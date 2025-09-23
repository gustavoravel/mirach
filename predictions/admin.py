from django.contrib import admin
from .models import PredictionModel, Prediction, PredictionResult


@admin.register(PredictionModel)
class PredictionModelAdmin(admin.ModelAdmin):
    list_display = ['name', 'algorithm_type', 'is_active', 'created_at']
    list_filter = ['algorithm_type', 'is_active', 'created_at']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at']
    ordering = ['name']


class PredictionResultInline(admin.TabularInline):
    model = PredictionResult
    extra = 0
    fields = ['timestamp', 'actual_value', 'predicted_value', 'confidence_interval_lower', 'confidence_interval_upper']
    readonly_fields = ['timestamp']


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ['name', 'project', 'dataset', 'prediction_model', 'status', 'prediction_horizon', 'created_at']
    list_filter = ['status', 'prediction_model__algorithm_type', 'created_at', 'project']
    search_fields = ['name', 'description', 'project__name', 'dataset__name', 'created_by__username']
    readonly_fields = ['created_at', 'completed_at']
    ordering = ['-created_at']
    inlines = [PredictionResultInline]
    
    fieldsets = (
        ('Informações Básicas', {
            'fields': ('name', 'description', 'project', 'dataset', 'prediction_model', 'created_by')
        }),
        ('Parâmetros da Previsão', {
            'fields': ('prediction_horizon', 'train_size', 'validation_size', 'test_size')
        }),
        ('Status e Resultados', {
            'fields': ('status', 'model_parameters', 'metrics', 'predictions_data')
        }),
        ('Processamento', {
            'fields': ('completed_at', 'error_message'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )


@admin.register(PredictionResult)
class PredictionResultAdmin(admin.ModelAdmin):
    list_display = ['prediction', 'timestamp', 'actual_value', 'predicted_value']
    list_filter = ['prediction__project', 'prediction__prediction_model', 'timestamp']
    search_fields = ['prediction__name', 'prediction__project__name']
    readonly_fields = ['timestamp']
    ordering = ['prediction', 'timestamp']