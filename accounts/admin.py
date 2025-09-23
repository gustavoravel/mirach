from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User
from .models import UserProfile, UserActivity


class UserProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    verbose_name_plural = 'Perfil'


class UserAdmin(BaseUserAdmin):
    inlines = (UserProfileInline,)
    list_display = ('username', 'email', 'first_name', 'last_name', 'is_staff', 'date_joined')
    list_filter = ('is_staff', 'is_superuser', 'is_active', 'date_joined')


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'company', 'job_title', 'timezone', 'created_at']
    list_filter = ['timezone', 'language', 'email_notifications', 'created_at']
    search_fields = ['user__username', 'user__email', 'user__first_name', 'user__last_name', 'company']
    readonly_fields = ['created_at', 'updated_at', 'last_login_ip']
    ordering = ['-created_at']
    
    fieldsets = (
        ('Informações Pessoais', {
            'fields': ('user', 'company', 'job_title', 'phone', 'bio', 'avatar')
        }),
        ('Configurações', {
            'fields': ('timezone', 'language', 'email_notifications')
        }),
        ('Metadados', {
            'fields': ('created_at', 'updated_at', 'last_login_ip'),
            'classes': ('collapse',)
        }),
    )


@admin.register(UserActivity)
class UserActivityAdmin(admin.ModelAdmin):
    list_display = ['user', 'activity_type', 'description', 'ip_address', 'created_at']
    list_filter = ['activity_type', 'created_at', 'user']
    search_fields = ['user__username', 'description']
    readonly_fields = ['created_at']
    ordering = ['-created_at']
    
    fieldsets = (
        ('Atividade', {
            'fields': ('user', 'activity_type', 'description')
        }),
        ('Informações Técnicas', {
            'fields': ('ip_address', 'user_agent'),
            'classes': ('collapse',)
        }),
        ('Timestamp', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )


# Re-register UserAdmin
admin.site.unregister(User)
admin.site.register(User, UserAdmin)