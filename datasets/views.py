from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Dataset, ColumnMapping
from django.http import JsonResponse
from projects.models import ProjectMembership, Project
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from accounts.models import Subscription


@login_required
def dataset_list(request):
    # Datasets de projetos onde o usuário é membro
    member_project_ids = ProjectMembership.objects.filter(user=request.user).values_list('project_id', flat=True)
    datasets = Dataset.objects.filter(project_id__in=member_project_ids)
    return render(request, 'datasets/list.html', {'datasets': datasets})


@login_required
def dataset_upload(request):
    if request.method == 'POST':
        project_id = request.POST.get('project')
        file = request.FILES.get('file')
        
        if project_id and file:
            from .utils import process_excel_file, detect_time_series_columns, validate_time_series_data
            
            project = get_object_or_404(Project, pk=project_id)
            # Permissão: editor/owner
            membership = ProjectMembership.objects.filter(project=project, user=request.user, role__in=['owner','editor']).exists()
            if not membership:
                messages.error(request, 'Você não tem permissão para enviar datasets neste projeto.')
                return redirect('datasets:list')

            # Enforce plan limits
            sub = Subscription.current_for(request.user)
            if sub and not sub.plan.is_enterprise:
                # Count datasets for this user across projects
                from .models import Dataset as DS
                total_datasets = DS.objects.filter(project__memberships__user=request.user).count()
                if total_datasets >= sub.plan.max_datasets:
                    messages.warning(request, 'Limite de datasets atingido no seu plano. Faça upgrade para continuar.')
                    return redirect('accounts:profile')
            
            # Processar o arquivo Excel
            result = process_excel_file(file)
            
            if result['success']:
                # Max rows per dataset
                if sub and result.get('total_rows') and result['total_rows'] > sub.plan.max_rows_per_dataset:
                    messages.warning(request, f"Seu plano suporta até {sub.plan.max_rows_per_dataset} linhas por dataset. Faça upgrade para continuar.")
                    return redirect('accounts:profile')
                dataset = Dataset.objects.create(
                    name=file.name,
                    project=project,
                    file=file,
                    uploaded_by=request.user,
                    total_rows=result['total_rows'],
                    total_columns=result['total_columns'],
                    column_names=result['column_names'],
                    status='uploaded'
                )
                # Audit
                from projects.models import AuditLog
                AuditLog.objects.create(project=project, user=request.user, action='dataset_upload', context={'dataset_id': dataset.id, 'name': dataset.name})

                # Criar sugestões de mapeamento de colunas
                suggestions = detect_time_series_columns(result['dataframe'])
                for col_name, col_type in suggestions.items():
                    ColumnMapping.objects.create(
                        dataset=dataset,
                        column_name=col_name,
                        column_type=col_type,
                        data_type=result['column_types'].get(col_name, 'unknown')
                    )
                
                # Validação básica automática (se encontrar timestamp/target sugeridos)
                suggested_timestamp = next((c for c, t in suggestions.items() if t == 'timestamp'), None)
                suggested_target = next((c for c, t in suggestions.items() if t == 'target'), None)
                if suggested_timestamp and suggested_target:
                    validation = validate_time_series_data(result['dataframe'].copy(), suggested_timestamp, suggested_target)
                    if not validation['valid']:
                        messages.warning(request, 'Dados carregados com avisos/erros. Ajuste o mapeamento de colunas.')
                    else:
                        dataset.status = 'processed'
                        dataset.save(update_fields=['status'])
                        messages.success(request, 'Dataset carregado e validado com sucesso!')
                else:
                    messages.info(request, 'Dataset carregado. Confirme o mapeamento de colunas para processar.')
                return redirect('datasets:detail', pk=dataset.pk)
            else:
                messages.error(request, f'Erro ao processar arquivo: {result["error"]}')
        else:
            messages.error(request, 'Projeto e arquivo são obrigatórios.')
    
    from projects.models import Project
    projects = Project.objects.filter(owner=request.user, is_active=True)
    return render(request, 'datasets/upload.html', {'projects': projects})


@login_required
def dataset_detail(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk)
    # Permissão: viewer+ (membro)
    if not ProjectMembership.objects.filter(project=dataset.project, user=request.user).exists():
        messages.error(request, 'Acesso negado a este dataset.')
        return redirect('datasets:list')
    return render(request, 'datasets/detail.html', {'dataset': dataset})


@login_required
def dataset_edit(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk)
    if not ProjectMembership.objects.filter(project=dataset.project, user=request.user, role__in=['owner','editor']).exists():
        messages.error(request, 'Você não tem permissão para editar este dataset.')
        return redirect('datasets:list')
    
    if request.method == 'POST':
        dataset.name = request.POST.get('name', dataset.name)
        dataset.description = request.POST.get('description', dataset.description)
        dataset.save()
        messages.success(request, 'Dataset atualizado com sucesso!')
        return redirect('datasets:detail', pk=dataset.pk)
    
    return render(request, 'datasets/edit.html', {'dataset': dataset})


@login_required
def dataset_delete(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk)
    if not ProjectMembership.objects.filter(project=dataset.project, user=request.user, role__in=['owner','editor']).exists():
        messages.error(request, 'Você não tem permissão para excluir este dataset.')
        return redirect('datasets:list')
    
    if request.method == 'POST':
        dataset.delete()
        messages.success(request, 'Dataset excluído com sucesso!')
        return redirect('datasets:list')
    
    return render(request, 'datasets/delete.html', {'dataset': dataset})


@login_required
def column_mapping(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk)
    if not ProjectMembership.objects.filter(project=dataset.project, user=request.user, role__in=['owner','editor']).exists():
        messages.error(request, 'Você não tem permissão para processar este dataset.')
        return redirect('datasets:list')
    
    if request.method == 'POST':
        # Process column mappings
        for key, value in request.POST.items():
            if key.startswith('column_type_'):
                column_name = key.replace('column_type_', '')
                mapping, created = ColumnMapping.objects.get_or_create(
                    dataset=dataset,
                    column_name=column_name
                )
                mapping.column_type = value
                mapping.save()
        
        messages.success(request, 'Mapeamento de colunas salvo com sucesso!')
        return redirect('datasets:detail', pk=dataset.pk)
    
    return render(request, 'datasets/mapping.html', {'dataset': dataset})


@login_required
def dataset_process(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk, project__owner=request.user)
    
    if request.method == 'POST':
        # TODO: Implement dataset processing logic
        dataset.status = 'processed'
        dataset.save()
        messages.success(request, 'Dataset processado com sucesso!')
        return redirect('datasets:detail', pk=dataset.pk)
    
    return render(request, 'datasets/process.html', {'dataset': dataset})


@login_required
def dataset_backtest(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk, project__owner=request.user)
    # Proxy to predictions backtest API
    from predictions.views import backtest
    request.GET._mutable = True
    if 'models' not in request.GET:
        request.GET.setlist('models', ['arima', 'ets', 'prophet'])
    return backtest(request, dataset_id=pk)


# Public API for datasets
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def api_list_datasets(request):
    member_project_ids = ProjectMembership.objects.filter(user=request.user).values_list('project_id', flat=True)
    qs = Dataset.objects.filter(project_id__in=member_project_ids).order_by('-uploaded_at')[:100]
    return JsonResponse({'datasets': [
        {'id': d.id, 'name': d.name, 'project_id': d.project_id, 'rows': d.total_rows, 'cols': d.total_columns}
        for d in qs
    ]})