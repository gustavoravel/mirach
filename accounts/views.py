from django.shortcuts import render, redirect
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import UserProfile
from .forms import EmailUserCreationForm


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
    return render(request, 'accounts/profile.html', {'profile': profile})


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