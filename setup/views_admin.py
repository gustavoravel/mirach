from django.contrib.admin.views.decorators import staff_member_required
from django.shortcuts import render
from django.db.models import Count, Avg, Q
from django.utils import timezone
from datetime import timedelta

from accounts.models import Subscription, Plan, UsageEvent
from projects.models import Project
from datasets.models import Dataset
from predictions.models import Prediction, PredictionResult


@staff_member_required
def saas_dashboard(request):
    now = timezone.now()
    rng = request.GET.get('range', '30d')
    days_map = {'7d': 7, '30d': 30, '90d': 90}
    days = days_map.get(rng, 30)
    last_24h = now - timedelta(hours=24)
    last_48h = now - timedelta(hours=48)
    last_nd = now - timedelta(days=days)

    # North Star: previsões concluídas por usuários ativos (últimos 30d)
    active_users = UsageEvent.objects.filter(created_at__gte=last_nd).values_list('user_id', flat=True).distinct()
    useful_predictions = Prediction.objects.filter(status='completed', created_by_id__in=active_users, completed_at__gte=last_nd).count()

    # Ativação: % usuários com 1 projeto, 1 dataset e 1 previsão em 48h após signup (aproximação via timestamps)
    total_users_48h = Subscription.objects.filter(started_at__gte=last_48h).values_list('user_id', flat=True).distinct().count()
    activated_48h = 0
    if total_users_48h:
        for user_id in Subscription.objects.filter(started_at__gte=last_48h).values_list('user_id', flat=True).distinct():
            has_project = Project.objects.filter(owner_id=user_id, created_at__gte=last_48h).exists()
            has_dataset = Dataset.objects.filter(uploaded_by_id=user_id, uploaded_at__gte=last_48h).exists()
            has_prediction = Prediction.objects.filter(created_by_id=user_id, created_at__gte=last_48h).exists()
            if has_project and has_dataset and has_prediction:
                activated_48h += 1
    activation_rate = (activated_48h / total_users_48h * 100) if total_users_48h else 0

    # Retenção: DAU/WAU por projeto (últimos 7d)
    dau = UsageEvent.objects.filter(created_at__gte=last_24h).values('user_id').distinct().count()
    wau = UsageEvent.objects.filter(created_at__gte=last_7d).values('user_id').distinct().count()

    # Qualidade: medianos (aprox por avg) de MAPE/RMSE (últimos 30d)
    recent_preds = Prediction.objects.filter(status='completed', completed_at__gte=last_nd)
    # métricas guardadas em JSON; exemplo simplificado
    avg_rmse = recent_preds.aggregate(v=Avg('metrics__rmse'))['v'] or 0
    avg_mape = recent_preds.aggregate(v=Avg('metrics__mape'))['v'] or 0

    # Unidade econômica (simplificada): contagem de previsões e saldo de créditos
    preds_nd = Prediction.objects.filter(created_at__gte=last_nd).count()

    # Top projetos por previsões no período
    top_projects = (
        Prediction.objects.filter(created_at__gte=last_nd)
        .values('project__id', 'project__name')
        .annotate(cnt=Count('id'))
        .order_by('-cnt')[:10]
    )

    # Distribuição de planos ativos
    plans_distribution = (
        Subscription.objects.filter(is_active=True)
        .values('plan__code', 'plan__name')
        .annotate(n=Count('id'))
        .order_by('-n')
    )

    context = {
        'useful_predictions': useful_predictions,
        'activation_rate_48h': activation_rate,
        'dau': dau,
        'wau': wau,
        'avg_rmse': round(avg_rmse, 2),
        'avg_mape': round(avg_mape, 2),
        'predictions_count': preds_nd,
        'range': rng,
        'top_projects': top_projects,
        'plans_distribution': plans_distribution,
    }
    return render(request, 'admin/saas_dashboard.html', context)


