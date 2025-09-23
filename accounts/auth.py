from rest_framework.authentication import BaseAuthentication
from rest_framework import exceptions
from .models import APIToken


class TokenAuthentication(BaseAuthentication):
    keyword = 'Token'

    def authenticate(self, request):
        auth = request.headers.get('Authorization') or ''
        parts = auth.split()
        if len(parts) != 2 or parts[0] != self.keyword:
            return None
        key = parts[1]
        try:
            token = APIToken.objects.select_related('user').get(key=key, is_active=True)
        except APIToken.DoesNotExist:
            raise exceptions.AuthenticationFailed('Invalid token')
        return (token.user, token)

