from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import os


def dataset_upload_path(instance, filename):
    """Função para definir o caminho de upload dos datasets"""
    return f'datasets/{instance.project.id}/{filename}'


class Dataset(models.Model):
    """Modelo para datasets de séries temporais"""
    
    FILE_STATUS_CHOICES = [
        ('uploaded', 'Carregado'),
        ('processing', 'Processando'),
        ('processed', 'Processado'),
        ('error', 'Erro'),
    ]
    
    name = models.CharField(max_length=200, verbose_name="Nome do Dataset")
    description = models.TextField(blank=True, verbose_name="Descrição")
    project = models.ForeignKey('projects.Project', on_delete=models.CASCADE, 
                               related_name='datasets', verbose_name="Projeto")
    file = models.FileField(upload_to=dataset_upload_path, verbose_name="Arquivo")
    file_size = models.BigIntegerField(null=True, blank=True, verbose_name="Tamanho do Arquivo")
    file_type = models.CharField(max_length=50, blank=True, verbose_name="Tipo do Arquivo")
    status = models.CharField(max_length=20, choices=FILE_STATUS_CHOICES, 
                             default='uploaded', verbose_name="Status")
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="Carregado por")
    uploaded_at = models.DateTimeField(default=timezone.now, verbose_name="Carregado em")
    processed_at = models.DateTimeField(null=True, blank=True, verbose_name="Processado em")
    error_message = models.TextField(blank=True, verbose_name="Mensagem de Erro")
    
    # Metadados do arquivo
    total_rows = models.IntegerField(null=True, blank=True, verbose_name="Total de Linhas")
    total_columns = models.IntegerField(null=True, blank=True, verbose_name="Total de Colunas")
    column_names = models.JSONField(default=list, blank=True, verbose_name="Nomes das Colunas")
    
    class Meta:
        verbose_name = "Dataset"
        verbose_name_plural = "Datasets"
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.name} ({self.project.name})"
    
    def save(self, *args, **kwargs):
        if self.file:
            self.file_size = self.file.size
            self.file_type = os.path.splitext(self.file.name)[1].lower()
        super().save(*args, **kwargs)


class ColumnMapping(models.Model):
    """Modelo para mapeamento de colunas do dataset para algoritmos"""
    
    COLUMN_TYPES = [
        ('timestamp', 'Timestamp/Data'),
        ('target', 'Variável Alvo'),
        ('feature', 'Variável Explicativa'),
        ('ignore', 'Ignorar'),
    ]
    
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, 
                               related_name='column_mappings', verbose_name="Dataset")
    column_name = models.CharField(max_length=100, verbose_name="Nome da Coluna")
    column_type = models.CharField(max_length=20, choices=COLUMN_TYPES, 
                                  verbose_name="Tipo da Coluna")
    is_required = models.BooleanField(default=False, verbose_name="Obrigatório")
    data_type = models.CharField(max_length=50, blank=True, verbose_name="Tipo de Dados")
    description = models.TextField(blank=True, verbose_name="Descrição")
    created_at = models.DateTimeField(default=timezone.now, verbose_name="Criado em")
    
    class Meta:
        verbose_name = "Mapeamento de Coluna"
        verbose_name_plural = "Mapeamentos de Colunas"
        unique_together = ['dataset', 'column_name']
    
    def __str__(self):
        return f"{self.dataset.name} - {self.column_name} ({self.column_type})"