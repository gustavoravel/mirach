from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Dataset, ColumnMapping
from django.http import JsonResponse
from projects.models import ProjectMembership, Project
from django.urls import reverse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from accounts.models import Subscription
import json
import pandas as pd
import numpy as np
from datetime import datetime, date


def sanitize_for_json(obj):
    """Convert non-JSON serializable objects to serializable ones"""
    if isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj


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
            from .utils import process_file, detect_time_series_columns, validate_time_series_data
            
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
            
            # Processar o arquivo (Excel/CSV)
            result = process_file(file)
            
            if result['success']:
                # Max rows per dataset
                if sub and result.get('total_rows') and result['total_rows'] > sub.plan.max_rows_per_dataset:
                    messages.warning(request, f"Seu plano suporta até {sub.plan.max_rows_per_dataset} linhas por dataset. Faça upgrade para continuar.")
                    return redirect('accounts:profile')
                try:
                    # Garantir ponteiro no início ao salvar o arquivo no FileField
                    if hasattr(file, 'seek'):
                        file.seek(0)
                except Exception:
                    pass
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
                else:
                    validation = {'valid': False, 'errors': ['Mapeie timestamp e alvo'], 'warnings': []}

                # Renderizar preview com primeiras linhas e informações detectadas
                request.session[f'ds_preview_{dataset.pk}'] = {
                    'detected': result.get('detected', {}),
                    'suggested_timestamp': suggested_timestamp,
                    'suggested_target': suggested_target,
                }
                return redirect('datasets:preview', pk=dataset.pk)
            else:
                messages.error(request, f'Erro ao processar arquivo: {result["error"]}')
        else:
            messages.error(request, 'Projeto e arquivo são obrigatórios.')
    
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
def dataset_preview(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk)
    if not ProjectMembership.objects.filter(project=dataset.project, user=request.user).exists():
        messages.error(request, 'Acesso negado a este dataset.')
        return redirect('datasets:list')
    # Gera preview a partir do arquivo salvo
    import pandas as pd
    detected = request.session.get(f'ds_preview_{pk}', {}).get('detected', {})
    suggested_timestamp = request.session.get(f'ds_preview_{pk}', {}).get('suggested_timestamp')
    suggested_target = request.session.get(f'ds_preview_{pk}', {}).get('suggested_target')
    try:
        if dataset.file.name.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(dataset.file.path)
        else:
            df = pd.read_csv(dataset.file.path, low_memory=False)
        preview_html = df.head(10).to_html(classes='table table-sm table-striped', index=False)
    except Exception:
        preview_html = '<div class="text-danger">Não foi possível gerar o preview.</div>'
    return render(request, 'datasets/preview.html', {
        'dataset': dataset,
        'preview_html': preview_html,
        'detected': detected,
        'validation': {'valid': True, 'errors': [], 'warnings': []},
        'suggested_timestamp': suggested_timestamp,
        'suggested_target': suggested_target,
    })


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
    dataset = get_object_or_404(Dataset, pk=pk)
    if not ProjectMembership.objects.filter(project=dataset.project, user=request.user).exists():
        messages.error(request, 'Acesso negado a este dataset.')
        return redirect('datasets:list')
    # Feature gate: somente planos pagos abrem backtest
    from accounts.models import Subscription
    sub = Subscription.current_for(request.user)
    if sub and sub.plan.code == 'free':
        return redirect(f"{reverse('datasets:detail', args=[pk])}?upgrade=1")
    # Proxy to predictions backtest API
    from predictions.views import backtest
    request.GET._mutable = True
    if 'models' not in request.GET:
        request.GET.setlist('models', ['arima', 'ets', 'prophet'])
    return backtest(request, dataset_id=pk)


@login_required
def dataset_explore(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk)
    if not ProjectMembership.objects.filter(project=dataset.project, user=request.user).exists():
        messages.error(request, 'Acesso negado a este dataset.')
        return redirect('datasets:list')
    import pandas as pd, io
    try:
        with dataset.file.open('rb') as fh:
            content = fh.read()
        if dataset.file.name.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content), low_memory=False)
    except Exception as e:
        messages.error(request, f'Falha ao abrir dataset: {e}')
        return redirect('datasets:detail', pk=pk)
    sample = df.head(200)
    info = {
        'rows': int(df.shape[0]),
        'cols': int(df.shape[1]),
        'columns': [{'name': c, 'dtype': str(df[c].dtype)} for c in df.columns]
    }
    desc = sample.describe(include='all').replace({pd.NA: ''}).astype(str).reset_index().values.tolist()
    return render(request, 'datasets/explore.html', {
        'dataset': dataset,
        'sample': sample.to_dict(orient='records'),
        'columns': list(sample.columns),
        'info': info,
        'desc': desc,
    })

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


@login_required
def dataset_explore_interactive(request, pk):
    """Interactive exploration page integrated in Django"""
    dataset = get_object_or_404(Dataset, pk=pk)
    
    # Check permissions
    if not ProjectMembership.objects.filter(project=dataset.project, user=request.user).exists():
        messages.error(request, 'Você não tem permissão para acessar este dataset.')
        return redirect('datasets:list')
    
    # Prepare data for the interactive page
    dataset_data = {
        'dataframe': [],
        'mappings': {},
        'column_names': dataset.column_names or [],
        'total_rows': dataset.total_rows or 0,
        'total_columns': dataset.total_columns or 0,
        'memory_usage': 0,
        'missing_data': 0,
        'missing_data_by_column': {},
        'correlation_matrix': {},
        'numerical_stats': {},
        'columns_with_missing': 0,
        'numerical_columns': 0,
        'categorical_columns': 0
    }
    
    # Load dataset file
    try:
        if dataset.file:
            # Read dataset file using file.open() for remote storage compatibility
            with dataset.file.open('rb') as file:
                if dataset.file.name.endswith('.xlsx'):
                    df = pd.read_excel(file)
                else:
                    df = pd.read_csv(file)
            
            # Get column mappings
            mappings = {}
            for mapping in dataset.column_mappings.all():
                mappings[mapping.column_name] = mapping.column_type
            
            dataset_data['mappings'] = mappings
            
            # Convert dataframe to list of dictionaries for JSON serialization
            dataframe_dict = df.head(1000).to_dict('records')  # Limit to 1000 rows for performance
            dataset_data['dataframe'] = sanitize_for_json(dataframe_dict)
            
            # Calculate statistics
            dataset_data['memory_usage'] = round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            
            # Missing data analysis
            missing_data = df.isnull().sum()
            dataset_data['missing_data'] = int(missing_data.sum())
            dataset_data['missing_data_by_column'] = {col: int(count) for col, count in missing_data.items() if count > 0}
            dataset_data['columns_with_missing'] = int((missing_data > 0).sum())
            
            # Column type analysis
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            dataset_data['numerical_columns'] = len(numerical_cols)
            dataset_data['categorical_columns'] = len(categorical_cols)
            
            # Correlation matrix for numerical columns
            if len(numerical_cols) > 1:
                corr_matrix = df[numerical_cols].corr()
                dataset_data['correlation_matrix'] = sanitize_for_json(corr_matrix.to_dict())
            
            # Numerical statistics
            if len(numerical_cols) > 0:
                stats_df = df[numerical_cols].describe()
                dataset_data['numerical_stats'] = sanitize_for_json(stats_df.to_dict())
                
    except Exception as e:
        print(f"Error loading dataset: {e}")
        messages.error(request, f'Erro ao carregar dados do dataset: {str(e)}')
    
    return render(request, 'datasets/explore_interactive.html', {
        'dataset': dataset,
        'dataset_data': json.dumps(sanitize_for_json(dataset_data))
    })