from django.contrib import admin
from .models import Project


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ['name', 'owner', 'created_at', 'is_active', 'datasets_count', 'predictions_count']
    list_filter = ['is_active', 'created_at', 'owner']
    search_fields = ['name', 'description', 'owner__username', 'owner__email']
    readonly_fields = ['created_at', 'updated_at', 'datasets_count', 'predictions_count']
    ordering = ['-created_at']
    
    fieldsets = (
        ('Informações Básicas', {
            'fields': ('name', 'description', 'owner')
        }),
        ('Status', {
            'fields': ('is_active',)
        }),
        ('Metadados', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
        ('Estatísticas', {
            'fields': ('datasets_count', 'predictions_count'),
            'classes': ('collapse',)
        }),
    )