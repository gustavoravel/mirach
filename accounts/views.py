from django.shortcuts import render, redirect
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import UserProfile
from .forms import EmailUserCreationForm
from .models import APIToken
from django.http import JsonResponse


def signup(request):
    if request.method == 'POST':
        form = EmailUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Create user profile
            UserProfile.objects.create(user=user)
            login(request, user)
            messages.success(request, 'Conta criada com sucesso!')
            return redirect('projects:list')
        else:
            # Exibir erros para facilitar o debug no front
            messages.error(request, form.errors.as_text())
    else:
        form = EmailUserCreationForm()
    return render(request, 'accounts/signup.html', {'form': form})


@login_required
def profile(request):
    try:
        profile = request.user.userprofile
    except UserProfile.DoesNotExist:
        profile = UserProfile.objects.create(user=request.user)
    tokens = APIToken.objects.filter(user=request.user, is_active=True).order_by('-created_at')
    return render(request, 'accounts/profile.html', {'profile': profile, 'tokens': tokens})


@login_required
def edit_profile(request):
    try:
        profile = request.user.userprofile
    except UserProfile.DoesNotExist:
        profile = UserProfile.objects.create(user=request.user)
    
    if request.method == 'POST':
        profile.company = request.POST.get('company', '')
        profile.job_title = request.POST.get('job_title', '')
        profile.phone = request.POST.get('phone', '')
        profile.bio = request.POST.get('bio', '')
        profile.save()
        messages.success(request, 'Perfil atualizado com sucesso!')
        return redirect('accounts:profile')
    
    return render(request, 'accounts/edit_profile.html', {'profile': profile})


@login_required
def create_api_token(request):
    if request.method == 'POST':
        name = request.POST.get('name', '')
        token = APIToken.objects.create(user=request.user, name=name)
        messages.success(request, 'Token criado com sucesso.')
        return redirect('accounts:profile')
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@login_required
def revoke_api_token(request, key):
    if request.method == 'POST':
        try:
            token = APIToken.objects.get(user=request.user, key=key, is_active=True)
            token.is_active = False
            token.save(update_fields=['is_active'])
            messages.success(request, 'Token revogado.')
        except APIToken.DoesNotExist:
            messages.error(request, 'Token não encontrado.')
        return redirect('accounts:profile')
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@login_required
def settings_view(request):
    try:
        profile = request.user.userprofile
    except UserProfile.DoesNotExist:
        profile = UserProfile.objects.create(user=request.user)
    if request.method == 'POST':
        profile.language = request.POST.get('language') or profile.language
        profile.timezone = request.POST.get('timezone') or profile.timezone
        profile.email_notifications = bool(request.POST.get('email_notifications'))
        profile.company = request.POST.get('company', profile.company)
        profile.job_title = request.POST.get('job_title', profile.job_title)
        profile.phone = request.POST.get('phone', profile.phone)
        profile.bio = request.POST.get('bio', profile.bio)
        profile.save()
        messages.success(request, 'Configurações salvas.')
        return redirect('accounts:settings')
    return render(request, 'accounts/settings.html', {'profile': profile})


@login_required
def billing_view(request):
    from .models import Subscription, Plan, CreditBalance
    sub = Subscription.current_for(request.user)
    # Garantir assinatura Free por padrão
    if not Subscription.objects.filter(user=request.user, is_active=True).exists():
        free = Plan.objects.filter(code='free').first()
        if free:
            sub = Subscription.objects.create(user=request.user, plan=free, is_active=True)
    credits = CreditBalance.objects.filter(user=request.user).first()
    if credits is None:
        credits = CreditBalance.objects.create(user=request.user, balance=0)
    plans = Plan.objects.all().order_by('is_enterprise', 'code')
    if request.method == 'POST':
        plan_code = request.POST.get('plan')
        plan = Plan.objects.filter(code=plan_code).first()
        if plan:
            # deactivate others
            Subscription.objects.filter(user=request.user, is_active=True).update(is_active=False)
            Subscription.objects.create(user=request.user, plan=plan, is_active=True)
            messages.success(request, f'Plano atualizado para {plan.name}.')
        else:
            messages.error(request, 'Plano inválido.')
        return redirect('accounts:billing')
    return render(request, 'accounts/billing.html', {'subscription': sub, 'credits': credits, 'plans': plans})