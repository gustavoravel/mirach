from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User


class EmailUserCreationForm(UserCreationForm):
    # Override username to not be required (UserCreationForm defines it by default)
    username = forms.CharField(required=False, widget=forms.HiddenInput(), initial='')
    email = forms.EmailField(required=True, label='E-mail')

    class Meta:
        model = User
        fields = ('email',)

    def clean_email(self):
        email = self.cleaned_data.get('email', '').strip().lower()
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError('Já existe uma conta com este e-mail.')
        return email

    def clean_username(self):
        # Use email as username for the default User model
        email = self.cleaned_data.get('email', '').strip().lower()
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        email = self.cleaned_data['email'].strip().lower()
        # Usar o e-mail como username para compatibilidade com User padrão
        user.username = email
        user.email = email
        if commit:
            user.save()
        return user


class EmailAuthenticationForm(AuthenticationForm):
    username = forms.EmailField(label='E-mail')

    def clean(self):
        # Mapear e-mail para username do usuário antes de autenticar
        cleaned = super().clean()
        email = cleaned.get('username')
        if email:
            try:
                user = User.objects.get(email__iexact=email)
                # Substituir pelo username real para o backend padrão
                self.cleaned_data['username'] = user.username
            except User.DoesNotExist:
                pass  # Deixar o backend lidar com credenciais inválidas
        return self.cleaned_data


