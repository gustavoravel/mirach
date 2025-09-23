from django.db import models
from django.contrib.auth.models import User
import secrets


class APIToken(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='api_tokens')
    name = models.CharField(max_length=100, blank=True)
    key = models.CharField(max_length=40, unique=True, db_index=True)
    scopes = models.JSONField(default=list, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_used_at = models.DateTimeField(null=True, blank=True)

    def save(self, *args, **kwargs):
        if not self.key:
            self.key = secrets.token_hex(20)
        return super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.user.username} - {self.name or self.key[:6]}"
from django.contrib.auth.models import User
from django.utils import timezone


class UserProfile(models.Model):
    """Modelo para perfil estendido do usuário"""
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, verbose_name="Usuário")
    company = models.CharField(max_length=200, blank=True, verbose_name="Empresa")
    job_title = models.CharField(max_length=100, blank=True, verbose_name="Cargo")
    phone = models.CharField(max_length=20, blank=True, verbose_name="Telefone")
    bio = models.TextField(blank=True, verbose_name="Biografia")
    avatar = models.ImageField(upload_to='avatars/', blank=True, null=True, verbose_name="Avatar")
    
    # Configurações do usuário
    timezone = models.CharField(max_length=50, default='America/Sao_Paulo', verbose_name="Fuso Horário")
    language = models.CharField(max_length=10, default='pt-br', verbose_name="Idioma")
    email_notifications = models.BooleanField(default=True, verbose_name="Notificações por Email")
    
    # Metadados
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Criado em")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Atualizado em")
    last_login_ip = models.GenericIPAddressField(null=True, blank=True, verbose_name="Último IP de Login")
    
    class Meta:
        verbose_name = "Perfil do Usuário"
        verbose_name_plural = "Perfis dos Usuários"
    
    def __str__(self):
        return f"Perfil de {self.user.get_full_name() or self.user.username}"
    
    @property
    def full_name(self):
        return self.user.get_full_name() or self.user.username
    
    @property
    def projects_count(self):
        return self.user.project_set.count()
    
    @property
    def predictions_count(self):
        return self.user.prediction_set.count()


class UserActivity(models.Model):
    """Modelo para rastrear atividades do usuário"""
    
    ACTIVITY_TYPES = [
        ('login', 'Login'),
        ('logout', 'Logout'),
        ('project_created', 'Projeto Criado'),
        ('dataset_uploaded', 'Dataset Carregado'),
        ('prediction_created', 'Previsão Criada'),
        ('prediction_completed', 'Previsão Concluída'),
        ('profile_updated', 'Perfil Atualizado'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="Usuário")
    activity_type = models.CharField(max_length=50, choices=ACTIVITY_TYPES, verbose_name="Tipo de Atividade")
    description = models.TextField(verbose_name="Descrição")
    ip_address = models.GenericIPAddressField(null=True, blank=True, verbose_name="Endereço IP")
    user_agent = models.TextField(blank=True, verbose_name="User Agent")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Criado em")
    
    class Meta:
        verbose_name = "Atividade do Usuário"
        verbose_name_plural = "Atividades dos Usuários"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.user.username} - {self.get_activity_type_display()}"