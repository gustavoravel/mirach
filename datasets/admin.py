from django.contrib import admin
from .models import Dataset, ColumnMapping


class ColumnMappingInline(admin.TabularInline):
    model = ColumnMapping
    extra = 0
    fields = ['column_name', 'column_type', 'is_required', 'data_type', 'description']


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ['name', 'project', 'file_type', 'status', 'total_rows', 'total_columns', 'uploaded_at']
    list_filter = ['status', 'file_type', 'uploaded_at', 'project']
    search_fields = ['name', 'description', 'project__name', 'uploaded_by__username']
    readonly_fields = ['file_size', 'uploaded_at', 'processed_at', 'total_rows', 'total_columns']
    ordering = ['-uploaded_at']
    inlines = [ColumnMappingInline]
    
    fieldsets = (
        ('Informações Básicas', {
            'fields': ('name', 'description', 'project', 'uploaded_by')
        }),
        ('Arquivo', {
            'fields': ('file', 'file_size', 'file_type', 'status')
        }),
        ('Metadados do Arquivo', {
            'fields': ('total_rows', 'total_columns', 'column_names'),
            'classes': ('collapse',)
        }),
        ('Processamento', {
            'fields': ('processed_at', 'error_message'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('uploaded_at',),
            'classes': ('collapse',)
        }),
    )


@admin.register(ColumnMapping)
class ColumnMappingAdmin(admin.ModelAdmin):
    list_display = ['dataset', 'column_name', 'column_type', 'is_required', 'data_type']
    list_filter = ['column_type', 'is_required', 'dataset__project']
    search_fields = ['column_name', 'dataset__name', 'dataset__project__name']
    ordering = ['dataset', 'column_name']