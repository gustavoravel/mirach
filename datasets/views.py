from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Dataset, ColumnMapping


@login_required
def dataset_list(request):
    datasets = Dataset.objects.filter(project__owner=request.user)
    return render(request, 'datasets/list.html', {'datasets': datasets})


@login_required
def dataset_upload(request):
    if request.method == 'POST':
        project_id = request.POST.get('project')
        file = request.FILES.get('file')
        
        if project_id and file:
            from projects.models import Project
            from .utils import process_excel_file, detect_time_series_columns
            
            project = get_object_or_404(Project, pk=project_id, owner=request.user)
            
            # Processar o arquivo Excel
            result = process_excel_file(file)
            
            if result['success']:
                dataset = Dataset.objects.create(
                    name=file.name,
                    project=project,
                    file=file,
                    uploaded_by=request.user,
                    total_rows=result['total_rows'],
                    total_columns=result['total_columns'],
                    column_names=result['column_names'],
                    status='processed'
                )
                
                # Criar sugestões de mapeamento de colunas
                suggestions = detect_time_series_columns(result['dataframe'])
                for col_name, col_type in suggestions.items():
                    ColumnMapping.objects.create(
                        dataset=dataset,
                        column_name=col_name,
                        column_type=col_type,
                        data_type=result['column_types'].get(col_name, 'unknown')
                    )
                
                messages.success(request, 'Dataset carregado e processado com sucesso!')
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
    dataset = get_object_or_404(Dataset, pk=pk, project__owner=request.user)
    return render(request, 'datasets/detail.html', {'dataset': dataset})


@login_required
def dataset_edit(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk, project__owner=request.user)
    
    if request.method == 'POST':
        dataset.name = request.POST.get('name', dataset.name)
        dataset.description = request.POST.get('description', dataset.description)
        dataset.save()
        messages.success(request, 'Dataset atualizado com sucesso!')
        return redirect('datasets:detail', pk=dataset.pk)
    
    return render(request, 'datasets/edit.html', {'dataset': dataset})


@login_required
def dataset_delete(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk, project__owner=request.user)
    
    if request.method == 'POST':
        dataset.delete()
        messages.success(request, 'Dataset excluído com sucesso!')
        return redirect('datasets:list')
    
    return render(request, 'datasets/delete.html', {'dataset': dataset})


@login_required
def column_mapping(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk, project__owner=request.user)
    
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