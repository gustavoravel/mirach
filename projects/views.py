from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Project


@login_required
def project_list(request):
    projects = Project.objects.filter(owner=request.user, is_active=True)
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
            messages.success(request, 'Projeto criado com sucesso!')
            return redirect('projects:detail', pk=project.pk)
        else:
            messages.error(request, 'Nome do projeto é obrigatório.')
    
    return render(request, 'projects/create.html')


@login_required
def project_detail(request, pk):
    project = get_object_or_404(Project, pk=pk, owner=request.user)
    return render(request, 'projects/detail.html', {'project': project})


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