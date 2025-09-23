from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Project, ProjectMembership, ProjectInvitation, AuditLog
from accounts.models import Subscription
from django.http import JsonResponse
from django.utils import timezone
from django.contrib.auth.models import User
from django.urls import reverse


@login_required
def project_list(request):
    projects = Project.objects.filter(memberships__user=request.user, is_active=True).distinct()
    return render(request, 'projects/list.html', {'projects': projects})


@login_required
def project_create(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description', '')
        
        if name:
            project = Project.objects.create(
                name=name,
                description=description,
                owner=request.user
            )
            ProjectMembership.objects.get_or_create(project=project, user=request.user, defaults={'role': 'owner'})
            messages.success(request, 'Projeto criado com sucesso!')
            return redirect('projects:detail', pk=project.pk)
        else:
            messages.error(request, 'Nome do projeto é obrigatório.')
    
    return render(request, 'projects/create.html')


@login_required
def project_detail(request, pk):
    project = get_object_or_404(Project, pk=pk)
    if not ProjectMembership.objects.filter(project=project, user=request.user).exists():
        messages.error(request, 'Acesso negado a este projeto.')
        return redirect('projects:list')
    members = ProjectMembership.objects.filter(project=project).select_related('user')
    invites = ProjectInvitation.objects.filter(project=project, is_active=True)
    owner_sub = Subscription.current_for(project.owner)
    can_collaborate = bool(owner_sub and owner_sub.plan.code != 'free')
    logs = AuditLog.objects.filter(project=project)[:50]
    return render(request, 'projects/detail.html', {'project': project, 'members': members, 'invites': invites, 'logs': logs, 'can_collaborate': can_collaborate})


@login_required
def project_edit(request, pk):
    project = get_object_or_404(Project, pk=pk, owner=request.user)
    
    if request.method == 'POST':
        project.name = request.POST.get('name', project.name)
        project.description = request.POST.get('description', project.description)
        project.save()
        messages.success(request, 'Projeto atualizado com sucesso!')
        return redirect('projects:detail', pk=project.pk)
    
    return render(request, 'projects/edit.html', {'project': project})


@login_required
def project_delete(request, pk):
    project = get_object_or_404(Project, pk=pk, owner=request.user)
    
    if request.method == 'POST':
        project.is_active = False
        project.save()
        messages.success(request, 'Projeto excluído com sucesso!')
        return redirect('projects:list')
    
    return render(request, 'projects/delete.html', {'project': project})


@login_required
def project_invite(request, pk):
    project = get_object_or_404(Project, pk=pk)
    if not ProjectMembership.objects.filter(project=project, user=request.user, role__in=['owner','editor']).exists():
        messages.error(request, 'Apenas owner/editor podem convidar membros.')
        return redirect('projects:detail', pk=pk)
    # Feature gate: only paid plans can collaborate
    owner_sub = Subscription.current_for(project.owner)
    if not owner_sub or owner_sub.plan.code == 'free':
        messages.warning(request, 'Colaboração disponível apenas em planos pagos. Faça upgrade para convidar membros.')
        return redirect('projects:detail', pk=pk)

    if request.method == 'POST':
        email = request.POST.get('email', '').strip().lower()
        role = request.POST.get('role', 'viewer')
        if not email:
            messages.error(request, 'E-mail é obrigatório.')
            return redirect('projects:detail', pk=pk)
        invite = ProjectInvitation.objects.create(project=project, email=email, role=role, invited_by=request.user)
        # Enviar e-mail
        from django.core.mail import send_mail
        from django.conf import settings
        accept_url = request.build_absolute_uri(reverse('projects:accept_invite', args=[invite.token]))
        send_mail(
            subject=f"Convite para o projeto {project.name}",
            message=f"Você foi convidado para o projeto {project.name}. Acesse: {accept_url}",
            from_email=getattr(settings, 'DEFAULT_FROM_EMAIL', 'no-reply@mirach.local'),
            recipient_list=[email],
            fail_silently=True
        )
        messages.success(request, 'Convite enviado!')
        return redirect('projects:detail', pk=pk)

    return redirect('projects:detail', pk=pk)


@login_required
def accept_invite(request, token):
    invite = get_object_or_404(ProjectInvitation, token=token, is_active=True)
    # Vincular usuário atual ao projeto
    ProjectMembership.objects.get_or_create(project=invite.project, user=request.user, defaults={'role': invite.role})
    invite.accepted_by = request.user
    invite.accepted_at = timezone.now()
    invite.is_active = False
    invite.save()
    messages.success(request, f'Você entrou no projeto {invite.project.name} como {invite.role}.')
    return redirect('projects:detail', pk=invite.project.pk)