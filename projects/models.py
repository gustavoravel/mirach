from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import uuid


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


class ProjectMembership(models.Model):
    ROLE_CHOICES = [
        ('owner', 'Owner'),
        ('editor', 'Editor'),
        ('viewer', 'Viewer'),
    ]
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='memberships')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='project_memberships')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='viewer')
    invited_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='invitations_sent')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('project', 'user')
        verbose_name = 'Project Membership'
        verbose_name_plural = 'Project Memberships'

    def __str__(self):
        return f"{self.user.username} @ {self.project.name} ({self.role})"


class ProjectInvitation(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='invitations')
    email = models.EmailField()
    role = models.CharField(max_length=10, choices=ProjectMembership.ROLE_CHOICES, default='viewer')
    token = models.CharField(max_length=64, default=lambda: uuid.uuid4().hex, unique=True)
    invited_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='project_invites_created')
    accepted_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='project_invites_accepted')
    created_at = models.DateTimeField(auto_now_add=True)
    accepted_at = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"Invite {self.email} to {self.project.name} ({self.role})"


class AuditLog(models.Model):
    ACTION_CHOICES = [
        ('dataset_upload', 'Dataset Upload'),
        ('dataset_delete', 'Dataset Delete'),
        ('prediction_create', 'Prediction Create'),
        ('prediction_run', 'Prediction Run'),
        ('prediction_delete', 'Prediction Delete'),
    ]
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='audit_logs')
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    action = models.CharField(max_length=50, choices=ACTION_CHOICES)
    context = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
