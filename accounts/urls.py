from django.urls import path, include
from django.contrib.auth import views as auth_views
from django.shortcuts import render
from . import views
from .forms import EmailAuthenticationForm

app_name = 'accounts'

def landing_page(request):
    if request.user.is_authenticated:
        from django.shortcuts import redirect
        return redirect('projects:list')
    return render(request, 'landing.html')

urlpatterns = [
    path('', landing_page, name='landing'),
    path('login/', auth_views.LoginView.as_view(template_name='accounts/login.html', authentication_form=EmailAuthenticationForm), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('signup/', views.signup, name='signup'),
    path('profile/', views.profile, name='profile'),
    path('profile/edit/', views.edit_profile, name='edit_profile'),
    # Password reset
    path('password-reset/', auth_views.PasswordResetView.as_view(
        template_name='accounts/password_reset.html',
        email_template_name='accounts/password_reset_email.html',
        subject_template_name='accounts/password_reset_subject.txt'
    ), name='password_reset'),
    path('password-reset/done/', auth_views.PasswordResetDoneView.as_view(
        template_name='accounts/password_reset_done.html'
    ), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(
        template_name='accounts/password_reset_confirm.html'
    ), name='password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(
        template_name='accounts/password_reset_complete.html'
    ), name='password_reset_complete'),
]
