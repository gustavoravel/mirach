from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json
from .models import Prediction, PredictionModel
from .services import PredictionService
from datasets.models import Dataset


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
        train_size = float(request.POST.get('train_size', 0.8))
        validation_size = float(request.POST.get('validation_size', 0.1))
        
        if all([project_id, dataset_id, model_id, name, prediction_horizon]):
            from projects.models import Project
            
            project = get_object_or_404(Project, pk=project_id, owner=request.user)
            dataset = get_object_or_404(Dataset, pk=dataset_id, project__owner=request.user)
            model = get_object_or_404(PredictionModel, pk=model_id)
            
            # Get model parameters
            model_parameters = model.parameters.copy()
            
            # Override with user parameters if provided
            for key, value in request.POST.items():
                if key.startswith('param_'):
                    param_name = key.replace('param_', '')
                    try:
                        # Try to convert to appropriate type
                        if value.isdigit():
                            model_parameters[param_name] = int(value)
                        elif value.replace('.', '').isdigit():
                            model_parameters[param_name] = float(value)
                        elif value.lower() in ['true', 'false']:
                            model_parameters[param_name] = value.lower() == 'true'
                        else:
                            model_parameters[param_name] = value
                    except:
                        model_parameters[param_name] = value
            
            prediction = Prediction.objects.create(
                name=name,
                project=project,
                dataset=dataset,
                prediction_model=model,
                prediction_horizon=int(prediction_horizon),
                train_size=train_size,
                validation_size=validation_size,
                test_size=1.0 - train_size - validation_size,
                model_parameters=model_parameters,
                created_by=request.user
            )
            messages.success(request, 'Previsão criada com sucesso!')
            return redirect('predictions:detail', pk=prediction.pk)
        else:
            messages.error(request, 'Todos os campos são obrigatórios.')
    
    from projects.models import Project
    
    projects = Project.objects.filter(owner=request.user, is_active=True)
    datasets = Dataset.objects.filter(project__owner=request.user, status='processed')
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
        service = PredictionService()
        result = service.run_prediction(prediction)
        
        if result['success']:
            messages.success(request, 'Previsão executada com sucesso!')
        else:
            messages.error(request, f'Erro na previsão: {result["error"]}')
        
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


@login_required
def get_model_recommendations(request, dataset_id):
    """Get model recommendations for a dataset"""
    try:
        dataset = get_object_or_404(Dataset, pk=dataset_id, project__owner=request.user)
        service = PredictionService()
        result = service.get_model_recommendations(dataset)
        
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@login_required
def compare_models(request, dataset_id):
    """Compare multiple models on a dataset"""
    if request.method == 'POST':
        try:
            dataset = get_object_or_404(Dataset, pk=dataset_id, project__owner=request.user)
            models = request.POST.getlist('models')
            train_size = float(request.POST.get('train_size', 0.8))
            
            service = PredictionService()
            result = service.compare_models(dataset, models, train_size)
            
            return JsonResponse(result)
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Method not allowed'})


@login_required
def get_visualization_data(request, pk):
    """Get visualization data for a prediction"""
    try:
        prediction = get_object_or_404(Prediction, pk=pk, project__owner=request.user)
        service = PredictionService()
        result = service.create_visualization_data(prediction)
        
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@login_required
def prediction_status(request, pk):
    """Get prediction status (for AJAX polling)"""
    try:
        prediction = get_object_or_404(Prediction, pk=pk, project__owner=request.user)
        return JsonResponse({
            'status': prediction.status,
            'progress': prediction.get_progress_display() if hasattr(prediction, 'get_progress_display') else 0,
            'error_message': prediction.error_message
        })
    except Exception as e:
        return JsonResponse({'error': str(e)})