from django.urls import path, include
from django.contrib.auth import views as auth_views
from django.shortcuts import render
from . import views

app_name = 'accounts'

def landing_page(request):
    if request.user.is_authenticated:
        from django.shortcuts import redirect
        return redirect('projects:list')
    return render(request, 'landing.html')

urlpatterns = [
    path('', landing_page, name='landing'),
    path('login/', auth_views.LoginView.as_view(template_name='accounts/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('signup/', views.signup, name='signup'),
    path('profile/', views.profile, name='profile'),
    path('profile/edit/', views.edit_profile, name='edit_profile'),
]
