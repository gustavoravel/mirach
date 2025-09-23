from django.db import models
from django.contrib.auth.models import User
import secrets
from django.utils import timezone


class Plan(models.Model):
    code = models.CharField(max_length=30, unique=True)
    name = models.CharField(max_length=100)
    is_enterprise = models.BooleanField(default=False)
    # Limits
    max_projects = models.IntegerField(default=1)
    max_datasets = models.IntegerField(default=2)
    max_rows_per_dataset = models.IntegerField(default=50000)
    monthly_predictions = models.IntegerField(default=5)
    priority = models.CharField(max_length=10, default='low')  # low|high
    includes_advanced_models = models.BooleanField(default=False)
    includes_backtesting = models.BooleanField(default=False)
    includes_exports = models.BooleanField(default=False)

    def __str__(self):
        return self.name


class Subscription(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='subscriptions')
    plan = models.ForeignKey(Plan, on_delete=models.PROTECT)
    started_at = models.DateTimeField(default=timezone.now)
    trial_ends_at = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True)

    def is_trial_active(self):
        return bool(self.trial_ends_at and self.trial_ends_at > timezone.now())

    @staticmethod
    def current_for(user: User):
        sub = Subscription.objects.filter(user=user, is_active=True).order_by('-started_at').first()
        if sub:
            return sub
        # default to Free plan if none
        free = Plan.objects.filter(code='free').first()
        if free:
            return Subscription(user=user, plan=free, started_at=timezone.now(), is_active=True)
        return None

    def __str__(self):
        return f"{self.user.username} -> {self.plan.code}"


class UsageEvent(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='usage_events')
    event_type = models.CharField(max_length=50)  # dataset_upload, prediction_create, backtest
    created_at = models.DateTimeField(auto_now_add=True)
    metadata = models.JSONField(default=dict, blank=True)


class CreditBalance(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='credit_balances')
    balance = models.IntegerField(default=0)
    updated_at = models.DateTimeField(auto_now=True)


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