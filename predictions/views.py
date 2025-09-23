from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Prediction, PredictionModel


@login_required
def prediction_list(request):
    predictions = Prediction.objects.filter(project__owner=request.user)
    return render(request, 'predictions/list.html', {'predictions': predictions})


@login_required
def prediction_create(request):
    if request.method == 'POST':
        project_id = request.POST.get('project')
        dataset_id = request.POST.get('dataset')
        model_id = request.POST.get('model')
        name = request.POST.get('name')
        prediction_horizon = request.POST.get('prediction_horizon')
        
        if all([project_id, dataset_id, model_id, name, prediction_horizon]):
            from projects.models import Project
            from datasets.models import Dataset
            
            project = get_object_or_404(Project, pk=project_id, owner=request.user)
            dataset = get_object_or_404(Dataset, pk=dataset_id, project__owner=request.user)
            model = get_object_or_404(PredictionModel, pk=model_id)
            
            prediction = Prediction.objects.create(
                name=name,
                project=project,
                dataset=dataset,
                prediction_model=model,
                prediction_horizon=int(prediction_horizon),
                created_by=request.user
            )
            messages.success(request, 'Previsão criada com sucesso!')
            return redirect('predictions:detail', pk=prediction.pk)
        else:
            messages.error(request, 'Todos os campos são obrigatórios.')
    
    from projects.models import Project
    from datasets.models import Dataset
    
    projects = Project.objects.filter(owner=request.user, is_active=True)
    datasets = Dataset.objects.filter(project__owner=request.user)
    models = PredictionModel.objects.filter(is_active=True)
    
    return render(request, 'predictions/create.html', {
        'projects': projects,
        'datasets': datasets,
        'models': models
    })


@login_required
def prediction_detail(request, pk):
    prediction = get_object_or_404(Prediction, pk=pk, project__owner=request.user)
    return render(request, 'predictions/detail.html', {'prediction': prediction})


@login_required
def prediction_run(request, pk):
    prediction = get_object_or_404(Prediction, pk=pk, project__owner=request.user)
    
    if request.method == 'POST':
        # TODO: Implement prediction running logic
        prediction.status = 'training'
        prediction.save()
        messages.success(request, 'Previsão iniciada!')
        return redirect('predictions:detail', pk=prediction.pk)
    
    return render(request, 'predictions/run.html', {'prediction': prediction})


@login_required
def prediction_results(request, pk):
    prediction = get_object_or_404(Prediction, pk=pk, project__owner=request.user)
    return render(request, 'predictions/results.html', {'prediction': prediction})


@login_required
def prediction_delete(request, pk):
    prediction = get_object_or_404(Prediction, pk=pk, project__owner=request.user)
    
    if request.method == 'POST':
        prediction.delete()
        messages.success(request, 'Previsão excluída com sucesso!')
        return redirect('predictions:list')
    
    return render(request, 'predictions/delete.html', {'prediction': prediction})


@login_required
def prediction_models(request):
    models = PredictionModel.objects.filter(is_active=True)
    return render(request, 'predictions/models.html', {'models': models})