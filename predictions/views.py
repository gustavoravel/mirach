from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.http import HttpResponse
from rest_framework.decorators import api_view, authentication_classes, permission_classes, throttle_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.throttling import UserRateThrottle
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json
import subprocess
import threading
import time
from .models import Prediction, PredictionModel
from .services import PredictionService
from .tasks import run_prediction_task
from datasets.models import Dataset
from projects.models import ProjectMembership
from .algorithms import create_forecaster
import numpy as np
import pandas as pd
from django.urls import reverse
from django.views.decorators.cache import cache_page
from accounts.models import Subscription, Plan, UsageEvent
from django.conf import settings

ALLOWED_FREE_MODELS = {'arima', 'ets', 'prophet', 'linear_regression'}

def _debug_log(message: str):
    if getattr(settings, 'DEBUG', False):
        try:
            print(f"[DEBUG] {message}")
        except Exception:
            pass

@login_required
def prediction_list(request):
    member_project_ids = ProjectMembership.objects.filter(user=request.user).values_list('project_id', flat=True)
    predictions = Prediction.objects.filter(project_id__in=member_project_ids)
    return render(request, 'predictions/list.html', {'predictions': predictions})


@login_required
def prediction_create(request):
    if request.method == 'POST':
        project_id = request.POST.get('project')
        dataset_id = request.POST.get('dataset')
        model_id = request.POST.get('model')
        frequency = request.POST.get('frequency', 'D')
        name = request.POST.get('name')
        prediction_horizon = request.POST.get('prediction_horizon')
        train_size = float(request.POST.get('train_size', 0.8))
        validation_size = float(request.POST.get('validation_size', 0.1))
        
        if all([project_id, dataset_id, model_id, name, prediction_horizon]):
            from projects.models import Project
            project = get_object_or_404(Project, pk=project_id)
            if not ProjectMembership.objects.filter(project=project, user=request.user, role__in=['owner','editor']).exists():
                messages.error(request, 'Você não tem permissão para criar previsões neste projeto.')
                return redirect('predictions:list')
            dataset = get_object_or_404(Dataset, pk=dataset_id, project=project)
            model = get_object_or_404(PredictionModel, pk=model_id)
            
            # Enforce plan limits
            sub = Subscription.current_for(request.user)
            if sub and not sub.plan.is_enterprise:
                # Monthly predictions count
                from django.utils import timezone
                from datetime import timedelta
                start_month = timezone.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                month_preds = Prediction.objects.filter(created_by=request.user, created_at__gte=start_month).count()
                if month_preds >= sub.plan.monthly_predictions:
                    messages.warning(request, 'Limite mensal de previsões atingido no seu plano. Faça upgrade para continuar.')
                    return redirect('accounts:profile')
                # Model gating for Free plan
                if sub.plan.code == 'free' and not getattr(settings, 'DEBUG', False) and model.algorithm_type not in ALLOWED_FREE_MODELS:
                    messages.warning(request, 'Este modelo é Pro. Faça upgrade para utilizá-lo.')
                    return redirect(f"{reverse('predictions:create')}?upgrade=1")

            # Get model parameters
            model_parameters = model.parameters.copy()
            # Store frequency in model parameters for downstream usage
            model_parameters['frequency'] = frequency
            
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
            UsageEvent.objects.create(user=request.user, event_type='prediction_create', metadata={'prediction_id': prediction.id})
            from projects.models import AuditLog
            AuditLog.objects.create(project=project, user=request.user, action='prediction_create', context={'prediction_id': prediction.id, 'name': prediction.name})
            messages.success(request, 'Previsão criada com sucesso!')
            return redirect('predictions:detail', pk=prediction.pk)
        else:
            messages.error(request, 'Todos os campos são obrigatórios.')
    
    from projects.models import Project
    
    projects = Project.objects.filter(memberships__user=request.user, is_active=True).distinct()
    datasets = Dataset.objects.filter(project__memberships__user=request.user, status__in=['processed','uploaded'])
    sub = Subscription.current_for(request.user)
    if sub and sub.plan.code == 'free' and not getattr(settings, 'DEBUG', False):
        models = PredictionModel.objects.filter(is_active=True, algorithm_type__in=ALLOWED_FREE_MODELS)
    else:
        models = PredictionModel.objects.filter(is_active=True)
    
    return render(request, 'predictions/create.html', {
        'projects': projects,
        'datasets': datasets,
        'models': models,
        'plan_code': (sub.plan.code if sub else 'free'),
        'allowed_free': ALLOWED_FREE_MODELS,
    })


@login_required
def prediction_detail(request, pk):
    prediction = get_object_or_404(Prediction, pk=pk)
    if not ProjectMembership.objects.filter(project=prediction.project, user=request.user).exists():
        messages.error(request, 'Acesso negado a esta previsão.')
        return redirect('predictions:list')
    from .domains import get_domain, resolve_dataset_domain
    domain = resolve_dataset_domain(prediction.dataset)
    expl = prediction.explainability if isinstance(prediction.explainability, dict) else {}
    if expl.get('domain_code'):
        domain = get_domain(expl.get('domain_code'))
    return render(request, 'predictions/detail.html', {
        'prediction': prediction,
        'domain': domain,
    })


@login_required
def prediction_run(request, pk):
    prediction = get_object_or_404(Prediction, pk=pk)
    if not ProjectMembership.objects.filter(project=prediction.project, user=request.user, role__in=['owner','editor']).exists():
        messages.error(request, 'Você não tem permissão para executar esta previsão.')
        return redirect('predictions:list')
    # Feature gate: somente planos pagos
    sub = Subscription.current_for(request.user)
    if sub and sub.plan.code == 'free' and not getattr(settings, 'DEBUG', False):
        # Permitir execução se o modelo for básico; bloquear se avançado
        if prediction.prediction_model.algorithm_type not in ALLOWED_FREE_MODELS:
            messages.warning(request, 'Este modelo é Pro. Faça upgrade para continuar.')
            return redirect(f"{reverse('predictions:detail', args=[prediction.pk])}?upgrade=1")
        # Aplicar limites do plano Free (soft caps)
        max_free_horizon = 30
        if prediction.prediction_horizon > max_free_horizon:
            prediction.prediction_horizon = max_free_horizon
            prediction.save(update_fields=['prediction_horizon'])
            messages.info(request, f'Horizonte ajustado para {max_free_horizon} períodos no plano Free.')
    
    if request.method == 'POST':
        # enqueue async job
        from datetime import timedelta, datetime
        prediction.status = 'queued'
        prediction.progress = 0
        prediction.error_message = ''
        prediction.metrics = {}
        prediction.predictions_data = {}
        # estimativa simples baseada no tamanho do dataset
        est_minutes = max(1, int(prediction.dataset.total_rows or 1000) // 5000)
        prediction.estimated_completion = datetime.now() + timedelta(minutes=est_minutes)
        prediction.save()
        queue_name = 'predictions_high' if getattr(prediction.project, 'priority', 0) >= 1 else 'predictions_default'
        run_prediction_task.apply_async(args=[prediction.id], queue=queue_name)
        from projects.models import AuditLog
        AuditLog.objects.create(project=prediction.project, user=request.user, action='prediction_run', context={'prediction_id': prediction.id})
        messages.success(request, 'Previsão enfileirada! Você será notificado ao concluir.')
        return redirect('predictions:detail', pk=prediction.pk)

    # GET: open confirmation as modal on the detail page
    return redirect(f"{reverse('predictions:detail', args=[prediction.pk])}?run=1")


@login_required
def prediction_results(request, pk):
    """Redirect to interactive exploration page"""
    prediction = get_object_or_404(Prediction, pk=pk)
    if not ProjectMembership.objects.filter(project=prediction.project, user=request.user).exists():
        messages.error(request, 'Acesso negado a esta previsão.')
        return redirect('predictions:list')
    # Redirect to the unified exploration page
    return redirect('predictions:explore_interactive', pk=pk)


@login_required
def prediction_explore(request, pk):
    prediction = get_object_or_404(Prediction, pk=pk)
    if not ProjectMembership.objects.filter(project=prediction.project, user=request.user).exists():
        messages.error(request, 'Acesso negado a esta previsão.')
        return redirect('predictions:list')
    # Preparar dados para exploração
    try:
        from .services import PredictionService
        service = PredictionService()
        viz = service.create_visualization_data(prediction)
        if not viz.get('success'):
            messages.error(request, viz.get('error','Falha ao preparar dados de visualização'))
            return redirect('predictions:detail', pk=pk)
        data = viz['data']
    except Exception as e:
        messages.error(request, str(e))
        return redirect('predictions:detail', pk=pk)
    return render(request, 'predictions/explore.html', {
        'prediction': prediction,
        'viz': data,
    })


@login_required
def prediction_wizard(request):
    """Multi-step wizard: 1) escolher projeto/dataset, 2) escolher modelo, 3) parâmetros & confirmar"""
    step = int(request.GET.get('step', request.POST.get('step', 1)))
    session_key = 'prediction_wizard'
    state = request.session.get(session_key, {})

    from projects.models import Project
    projects = Project.objects.filter(owner=request.user, is_active=True)

    if request.method == 'POST':
        step = int(request.POST.get('step', step))
        if step == 1:
            state['project'] = int(request.POST.get('project'))
            state['dataset'] = int(request.POST.get('dataset'))
            request.session[session_key] = state
            return redirect(f"{request.path}?step=2")
        elif step == 2:
            auto_mode = request.POST.get('auto_mode') == '1'
            if auto_mode:
                dataset_id = state.get('dataset')
                try:
                    from .llm.orchestrator_agent import plan_model
                    plan = plan_model(int(dataset_id))
                    algo = plan.get('algorithm') or 'arima'
                    # Map algo name to PredictionModel
                    model_obj = PredictionModel.objects.filter(
                        is_active=True, algorithm_type=algo
                    ).first()
                    if not model_obj:
                        # Fallback aliases
                        aliases = {
                            'ridge': 'ridge',
                            'lasso': 'lasso',
                            'linear_regression': 'linear_regression',
                        }
                        model_obj = PredictionModel.objects.filter(
                            is_active=True, algorithm_type=aliases.get(algo, 'arima')
                        ).first()
                    if not model_obj:
                        messages.error(request, 'Nenhum modelo disponível para o plano Auto.')
                        return redirect(f"{request.path}?step=2")
                    state['model'] = model_obj.id
                    state['auto_plan'] = {
                        'algorithm': algo,
                        'parameters': plan.get('parameters') or {},
                        'rationale': plan.get('rationale') or '',
                        'beats_baseline': plan.get('beats_baseline'),
                        'championship': plan.get('_championship'),
                    }
                    request.session[session_key] = state
                    messages.info(
                        request,
                        f"Modo Auto selecionou {model_obj.name}"
                        + (f": {plan.get('rationale')}" if plan.get('rationale') else ''),
                    )
                    return redirect(f"{request.path}?step=3")
                except Exception as exc:
                    messages.error(request, f'Falha no modo Auto: {exc}')
                    return redirect(f"{request.path}?step=2")
            state['model'] = int(request.POST.get('model'))
            state.pop('auto_plan', None)
            request.session[session_key] = state
            return redirect(f"{request.path}?step=3")
        elif step == 3:
            # finalize
            train_size = float(request.POST.get('train_size', 0.8))
            validation_size = float(request.POST.get('validation_size', 0.1))
            horizon = int(request.POST.get('prediction_horizon', 12))
            name = request.POST.get('name', 'Nova Previsão')
            params_json = request.POST.get('params_json', '').strip()
            domain_code = (request.POST.get('domain_code') or '').strip()

            project = get_object_or_404(Project, pk=state['project'], owner=request.user)
            dataset = get_object_or_404(Dataset, pk=state['dataset'], project__owner=request.user)
            model = get_object_or_404(PredictionModel, pk=state['model'])
            model_parameters = model.parameters.copy()
            # Apply Auto plan parameters first
            auto_plan = state.get('auto_plan') or {}
            if auto_plan.get('parameters'):
                model_parameters.update(auto_plan['parameters'])
            if params_json:
                try:
                    overrides = json.loads(params_json)
                    if isinstance(overrides, dict):
                        model_parameters.update(overrides)
                except Exception:
                    messages.warning(request, 'Hiperparâmetros inválidos (JSON). Usando padrões do modelo.')

            from .domains import enrich_interpretation_with_domain, resolve_dataset_domain
            if domain_code:
                interp = dict(dataset.ai_interpretation or {})
                interp['domain_code'] = domain_code
                interp['inferred_domain'] = domain_code
                dataset.ai_interpretation = enrich_interpretation_with_domain(interp)
                dataset.save(update_fields=['ai_interpretation'])
            domain_meta = resolve_dataset_domain(dataset)

            explainability = {
                'domain_code': domain_meta['code'],
                'domain_label': domain_meta['label'],
            }
            if auto_plan:
                explainability['auto_plan'] = {
                    'rationale': auto_plan.get('rationale'),
                    'beats_baseline': auto_plan.get('beats_baseline'),
                }
            prediction = Prediction.objects.create(
                name=name,
                project=project,
                dataset=dataset,
                prediction_model=model,
                prediction_horizon=horizon,
                train_size=train_size,
                validation_size=validation_size,
                test_size=1.0 - train_size - validation_size,
                model_parameters=model_parameters,
                created_by=request.user,
                explainability=explainability,
            )
            # Store championship summary for narrative
            if auto_plan.get('championship'):
                expl = dict(prediction.explainability or {})
                expl['beats_baseline'] = auto_plan.get('beats_baseline')
                expl['championship'] = {
                    'best_model': auto_plan.get('algorithm'),
                    'beats_baseline': auto_plan.get('beats_baseline'),
                }
                prediction.explainability = expl
                prediction.save(update_fields=['explainability'])
            request.session.pop(session_key, None)
            messages.success(request, 'Previsão criada! Agora execute para gerar os resultados.')
            return redirect('predictions:detail', pk=prediction.pk)

    datasets = Dataset.objects.filter(project__owner=request.user)
    sub = Subscription.current_for(request.user)
    if sub and sub.plan.code == 'free' and not getattr(settings, 'DEBUG', False):
        models = PredictionModel.objects.filter(is_active=True, algorithm_type__in=ALLOWED_FREE_MODELS)
    else:
        models = PredictionModel.objects.filter(is_active=True)

    from .domains import domain_presets, resolve_dataset_domain
    selected_domain = None
    selected_domain_params = {}
    presets = domain_presets()
    if state.get('dataset'):
        try:
            ds = Dataset.objects.get(pk=state['dataset'], project__owner=request.user)
            selected_domain = resolve_dataset_domain(ds)
            selected_domain_params = (selected_domain.get('preset') or {}).get('parameters') or {}
        except Dataset.DoesNotExist:
            selected_domain = None

    context = {
        'step': step,
        'projects': projects,
        'datasets': datasets,
        'models': models,
        'state': state,
        'domain_presets': presets,
        'selected_domain': selected_domain,
        'domain_presets_data': presets,
        'selected_domain_params': selected_domain_params,
    }
    context.update({'plan_code': (sub.plan.code if sub else 'free'), 'allowed_free': ALLOWED_FREE_MODELS})
    return render(request, 'predictions/wizard.html', context)


@login_required
def prediction_delete(request, pk):
    prediction = get_object_or_404(Prediction, pk=pk)
    if not ProjectMembership.objects.filter(project=prediction.project, user=request.user, role__in=['owner','editor']).exists():
        messages.error(request, 'Você não tem permissão para excluir esta previsão.')
        return redirect('predictions:list')
    
    if request.method == 'POST':
        from projects.models import AuditLog
        AuditLog.objects.create(project=prediction.project, user=request.user, action='prediction_delete', context={'prediction_id': prediction.id})
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
        dataset = get_object_or_404(Dataset, pk=dataset_id)
        if not ProjectMembership.objects.filter(project=dataset.project, user=request.user).exists():
            return JsonResponse({'success': False, 'error': 'Acesso negado'})
        service = PredictionService()
        result = service.get_model_recommendations(dataset)
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@login_required
def api_auto_plan(request, dataset_id):
    """Run OrchestratorAgent / championship and return ModelPlan JSON."""
    try:
        dataset = get_object_or_404(Dataset, pk=dataset_id)
        if not ProjectMembership.objects.filter(project=dataset.project, user=request.user).exists():
            return JsonResponse({'success': False, 'error': 'Acesso negado'})
        async_mode = request.GET.get('async') == '1'
        if async_mode:
            from .tasks import run_auto_model_plan_task
            async_result = run_auto_model_plan_task.delay(dataset_id)
            return JsonResponse({'success': True, 'task_id': async_result.id})
        from .llm.orchestrator_agent import plan_model
        plan = plan_model(dataset_id)
        return JsonResponse({'success': True, 'plan': plan})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@login_required
def compare_models(request, dataset_id):
    """Compare multiple models on a dataset"""
    if request.method == 'POST':
        try:
            dataset = get_object_or_404(Dataset, pk=dataset_id)
            if not ProjectMembership.objects.filter(project=dataset.project, user=request.user).exists():
                return JsonResponse({'success': False, 'error': 'Acesso negado'})
            sub = Subscription.current_for(request.user)
            if sub and sub.plan.code == 'free' and not getattr(settings, 'DEBUG', False):
                return JsonResponse({'success': False, 'error': 'upgrade_required'}, status=403)
            models = request.POST.getlist('models')
            train_size = float(request.POST.get('train_size', 0.8))
            
            service = PredictionService()
            result = service.compare_models(dataset, models, train_size)
            
            return JsonResponse(result)
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Method not allowed'})


@login_required
def backtest(request, dataset_id):
    """Run a simple rolling-origin backtest and return JSON metrics"""
    try:
        dataset = get_object_or_404(Dataset, pk=dataset_id)
        if not ProjectMembership.objects.filter(project=dataset.project, user=request.user).exists():
            return JsonResponse({'success': False, 'error': 'Acesso negado'})
        sub = Subscription.current_for(request.user)
        if sub and sub.plan.code == 'free' and not getattr(settings, 'DEBUG', False):
            return JsonResponse({'success': False, 'error': 'upgrade_required'}, status=403)
        models = request.GET.getlist('models') or ['arima', 'ets', 'prophet']
        train_size = float(request.GET.get('train_size', 0.7))
        horizon = int(request.GET.get('horizon', 6))

        service = PredictionService()
        target_series, _ = service.prepare_data(dataset)
        n = len(target_series)
        train_end = int(n * train_size)
        results = {}

        for model_type in models:
            try:
                forecaster = create_forecaster(model_type)
            except Exception as e:
                results[model_type] = {'error': str(e)}
                continue
            errors = []
            for start in range(train_end, n - horizon, horizon):
                train = target_series[:start]
                test = target_series[start:start + horizon]
                try:
                    forecaster.fit(train)
                    preds = forecaster.predict(len(test))
                    mae = float(np.abs(np.array(test) - np.array(preds)).mean())
                    errors.append(mae)
                except Exception as e:
                    errors.append(None)
            errors = [e for e in errors if e is not None]
            results[model_type] = {
                'windows': len(errors),
                'mae_mean': float(np.mean(errors)) if errors else None,
            }

        return JsonResponse({'success': True, 'backtest': results})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@login_required
def get_visualization_data(request, pk):
    """Get visualization data for a prediction"""
    try:
        from django.core.cache import cache
        prediction = get_object_or_404(Prediction, pk=pk)
        if not ProjectMembership.objects.filter(project=prediction.project, user=request.user).exists():
            return JsonResponse({'error': 'Acesso negado'})
        cache_key = f"api:viz:{request.user.id}:{prediction.id}:{prediction.status}:{prediction.completed_at.isoformat() if prediction.completed_at else 'na'}"
        cached = cache.get(cache_key)
        if cached:
            return JsonResponse(cached)
        service = PredictionService()
        result = service.create_visualization_data(prediction)
        cache.set(cache_key, result, 60)
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@login_required
def prediction_status(request, pk):
    """Get prediction status (for AJAX polling)"""
    try:
        prediction = get_object_or_404(Prediction, pk=pk)
        if not ProjectMembership.objects.filter(project=prediction.project, user=request.user).exists():
            return JsonResponse({'error': 'Acesso negado'})
        payload = {
            'status': prediction.status,
            'progress': prediction.progress if hasattr(prediction, 'progress') else 0,
            'eta': prediction.estimated_completion.isoformat() if prediction.estimated_completion else None,
            'error_message': prediction.error_message
        }
        # Verbose fields only in DEBUG and when requested
        if getattr(settings, 'DEBUG', False) and request.GET.get('verbose'):
            payload.update({
                'id': prediction.id,
                'name': prediction.name,
                'project_id': prediction.project_id,
                'dataset_id': prediction.dataset_id,
                'model': getattr(prediction.prediction_model, 'algorithm_type', None),
                'created_at': prediction.created_at.isoformat() if prediction.created_at else None,
                'completed_at': prediction.completed_at.isoformat() if prediction.completed_at else None,
            })
        _debug_log(f"prediction_status user={request.user.id} pk={pk} payload={payload}")
        return JsonResponse(payload)
    except Exception as e:
        return JsonResponse({'error': str(e)})


@login_required
def export_results_csv(request, pk):
    prediction = get_object_or_404(Prediction, pk=pk)
    if not ProjectMembership.objects.filter(project=prediction.project, user=request.user).exists():
        return JsonResponse({'error': 'Acesso negado'})
    import csv
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="prediction_{pk}_results.csv"'
    writer = csv.writer(response)
    writer.writerow(['timestamp', 'predicted_value', 'actual_value', 'ci_lower', 'ci_upper'])
    for r in prediction.results.all():
        writer.writerow([r.timestamp, r.predicted_value, r.actual_value or '', r.confidence_interval_lower or '', r.confidence_interval_upper or ''])
    return response


@login_required
def export_results_json(request, pk):
    prediction = get_object_or_404(Prediction, pk=pk)
    if not ProjectMembership.objects.filter(project=prediction.project, user=request.user).exists():
        return JsonResponse({'error': 'Acesso negado'})
    data = [
        {
            'timestamp': r.timestamp.isoformat(),
            'predicted_value': r.predicted_value,
            'actual_value': r.actual_value,
            'ci_lower': r.confidence_interval_lower,
            'ci_upper': r.confidence_interval_upper,
        }
        for r in prediction.results.all()
    ]
    return JsonResponse({'results': data})


# Public API (Token auth)
class APIDefaultThrottle(UserRateThrottle):
    scope = 'user'


@api_view(['GET'])
@authentication_classes([])
@permission_classes([])
def api_ping(request):
    return JsonResponse({'pong': True})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
@throttle_classes([APIDefaultThrottle])
def api_list_predictions(request):
    from django.core.cache import cache
    cache_key = f"api:preds:{request.user.id}"
    data = cache.get(cache_key)
    if not data:
        member_project_ids = ProjectMembership.objects.filter(user=request.user).values_list('project_id', flat=True)
        qs = Prediction.objects.filter(project_id__in=member_project_ids).order_by('-created_at')[:100]
        data = {'predictions': [
            {'id': p.id, 'name': p.name, 'status': p.status, 'project_id': p.project_id}
            for p in qs
        ]}
        cache.set(cache_key, data, 30)  # 30s
    return JsonResponse(data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
@throttle_classes([APIDefaultThrottle])
def api_prediction_detail(request, pk):
    p = get_object_or_404(Prediction, pk=pk)
    if not ProjectMembership.objects.filter(project=p.project, user=request.user).exists():
        return JsonResponse({'error': 'forbidden'}, status=403)
    from django.core.cache import cache
    cache_key = f"api:pred:{request.user.id}:{p.id}"
    cached = cache.get(cache_key)
    if cached:
        return JsonResponse(cached)
    payload = {
        'id': p.id,
        'name': p.name,
        'status': p.status,
        'metrics': p.metrics,
        'explainability': p.explainability,
        'model_version': p.model_version,
        'dataset_version': p.dataset_version,
        'results_url_csv': request.build_absolute_uri(reverse('predictions:export_csv', args=[p.id])),
        'results_url_json': request.build_absolute_uri(reverse('predictions:export_json', args=[p.id])),
    }
    cache.set(cache_key, payload, 60)  # 60s
    return JsonResponse(payload)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@throttle_classes([APIDefaultThrottle])
def api_create_prediction(request):
    try:
        dataset_id = request.data.get('dataset_id')
        model_id = request.data.get('model_id')
        model_type = request.data.get('model_type')
        name = request.data.get('name') or 'API Prediction'
        horizon = int(request.data.get('prediction_horizon') or 12)
        train_size = float(request.data.get('train_size') or 0.8)
        validation_size = float(request.data.get('validation_size') or 0.1)
        params = request.data.get('params') or {}

        if not dataset_id:
            return JsonResponse({'error': 'dataset_id is required'}, status=400)
        dataset = get_object_or_404(Dataset, pk=dataset_id)
        if not ProjectMembership.objects.filter(project=dataset.project, user=request.user, role__in=['owner','editor']).exists():
            return JsonResponse({'error': 'forbidden'}, status=403)
        if model_id:
            model = get_object_or_404(PredictionModel, pk=model_id)
        elif model_type:
            model = get_object_or_404(PredictionModel, algorithm_type=model_type, is_active=True)
        else:
            return JsonResponse({'error': 'model_id or model_type is required'}, status=400)

        # Model gating for Free plan
        sub = Subscription.current_for(request.user)
        if sub and sub.plan.code == 'free' and not getattr(settings, 'DEBUG', False) and model.algorithm_type not in ALLOWED_FREE_MODELS:
            return JsonResponse({'error': 'upgrade_required', 'detail': 'Model not available for Free plan'}, status=403)

        model_parameters = model.parameters.copy()
        if isinstance(params, dict):
            model_parameters.update(params)

        prediction = Prediction.objects.create(
            name=name,
            project=dataset.project,
            dataset=dataset,
            prediction_model=model,
            prediction_horizon=horizon,
            train_size=train_size,
            validation_size=validation_size,
            test_size=1.0 - train_size - validation_size,
            model_parameters=model_parameters,
            created_by=request.user
        )
        run_prediction_task.delay(prediction.id)
        return JsonResponse({
            'success': True,
            'prediction_id': prediction.id,
            'status_url': request.build_absolute_uri(reverse('predictions:api_prediction_detail', args=[prediction.id]))
        }, status=201)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
@throttle_classes([APIDefaultThrottle])
def api_prediction_results(request, pk):
    p = get_object_or_404(Prediction, pk=pk)
    if not ProjectMembership.objects.filter(project=p.project, user=request.user).exists():
        return JsonResponse({'error': 'forbidden'}, status=403)
    data = [
        {
            'timestamp': r.timestamp.isoformat(),
            'predicted_value': r.predicted_value,
            'actual_value': r.actual_value,
            'ci_lower': r.confidence_interval_lower,
            'ci_upper': r.confidence_interval_upper,
        }
        for r in p.results.all()
    ]
    return JsonResponse({'prediction_id': p.id, 'results': data})


@login_required
def prediction_explore_interactive(request, pk):
    """Interactive exploration page integrated in Django"""
    prediction = get_object_or_404(Prediction, pk=pk)
    
    # Check permissions
    if not ProjectMembership.objects.filter(project=prediction.project, user=request.user).exists():
        messages.error(request, 'Você não tem permissão para acessar esta previsão.')
        return redirect('predictions:list')
    
    # Check if prediction is completed
    if prediction.status != 'completed':
        messages.warning(request, 'A previsão ainda não foi concluída.')
        return redirect('predictions:detail', pk=pk)

    explain = prediction.explainability if isinstance(prediction.explainability, dict) else {}
    feature_importance = (
        explain.get('feature_importances')
        or explain.get('feature_importance')
        or (prediction.model_parameters or {}).get('feature_importance')
        or {}
    )
    if isinstance(feature_importance, list):
        # [{"feature": "...", "importance": 0.1}, ...] or parallel lists
        try:
            feature_importance = {
                str(item.get('feature', item.get('name', i))): float(item.get('importance', item.get('value', 0)))
                for i, item in enumerate(feature_importance)
                if isinstance(item, dict)
            }
        except Exception:
            feature_importance = {}
    
    # Prepare data for the interactive page
    from .domains import get_domain, resolve_dataset_domain
    domain_meta = resolve_dataset_domain(prediction.dataset)
    if explain.get('domain_code'):
        domain_meta = get_domain(explain.get('domain_code'))

    prediction_data = {
        'historical_data': [],
        'results': [],
        'metrics': prediction.metrics or {},
        'model_algorithm': prediction.prediction_model.algorithm_type,
        'model_name': prediction.prediction_model.name,
        'model_parameters': prediction.model_parameters or {},
        'feature_importance': feature_importance if isinstance(feature_importance, dict) else {},
        'ai_insights': explain.get('ai_insights') or {},
        'prediction_horizon': prediction.prediction_horizon,
        'prediction_name': prediction.name,
        'domain': {
            'code': domain_meta.get('code'),
            'label': domain_meta.get('label'),
            'description': domain_meta.get('description'),
            'target_vocabulary': domain_meta.get('target_vocabulary'),
            'forecast_noun': domain_meta.get('forecast_noun'),
            'result_tips': domain_meta.get('result_tips') or [],
        },
    }
    
    # Load historical data from dataset
    try:
        dataset = prediction.dataset
        if dataset.file:
            # Get target column
            target_mapping = dataset.column_mappings.filter(column_type='target').first()
            target_col = target_mapping.column_name if target_mapping else None
            
            # Get timestamp column
            timestamp_mapping = dataset.column_mappings.filter(column_type='timestamp').first()
            timestamp_col = timestamp_mapping.column_name if timestamp_mapping else None
            
            if target_col and timestamp_col:
                # Read dataset file (supports remote storage)
                with dataset.file.open('rb') as fh:
                    if dataset.file.name.lower().endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(fh)
                    else:
                        from datasets.utils import _read_csv_bytes
                        try:
                            df, _meta = _read_csv_bytes(fh.read())
                        except Exception:
                            fh.seek(0)
                            df = pd.read_csv(fh)
                
                # Prepare historical data
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
                df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
                historical_df = df[[timestamp_col, target_col]].dropna()
                historical_df = historical_df.sort_values(timestamp_col)
                
                prediction_data['historical_data'] = [
                    {
                        'timestamp': row[timestamp_col].isoformat(),
                        'value': float(row[target_col])
                    }
                    for _, row in historical_df.iterrows()
                ]
                prediction_data['target_column'] = target_col
                prediction_data['timestamp_column'] = timestamp_col
    except Exception as e:
        logger = __import__('logging').getLogger(__name__)
        logger.exception("Error loading historical data for explore: %s", e)
    
    # Load prediction results
    try:
        results = prediction.results.all().order_by('timestamp')
        rows = []
        for result in results:
            predicted = float(result.predicted_value)
            actual = float(result.actual_value) if result.actual_value is not None else None
            ci_lower = float(result.confidence_interval_lower) if result.confidence_interval_lower is not None else None
            ci_upper = float(result.confidence_interval_upper) if result.confidence_interval_upper is not None else None
            error = (predicted - actual) if actual is not None else None
            abs_error = abs(error) if error is not None else None
            pct_error = (abs_error / abs(actual) * 100.0) if (actual is not None and actual != 0 and abs_error is not None) else None
            rows.append({
                'timestamp': result.timestamp.isoformat(),
                'predicted_value': predicted,
                'actual_value': actual,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'error': error,
                'abs_error': abs_error,
                'pct_error': pct_error,
            })
        prediction_data['results'] = rows
    except Exception as e:
        logger = __import__('logging').getLogger(__name__)
        logger.exception("Error loading prediction results for explore: %s", e)
    
    return render(request, 'predictions/explore_interactive.html', {
        'prediction': prediction,
        'prediction_data': prediction_data,
        'domain': domain_meta,
    })