from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone


class Project(models.Model):
    """Modelo para projetos de previsão de séries temporais"""
    
    name = models.CharField(max_length=200, verbose_name="Nome do Projeto")
    description = models.TextField(blank=True, verbose_name="Descrição")
    owner = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="Proprietário")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Criado em")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Atualizado em")
    is_active = models.BooleanField(default=True, verbose_name="Ativo")
    
    class Meta:
        verbose_name = "Projeto"
        verbose_name_plural = "Projetos"
        ordering = ['-created_at']
    
    def __str__(self):
        return self.name
    
    @property
    def datasets_count(self):
        return self.datasets.count()
    
    @property
    def predictions_count(self):
        return self.predictions.count()