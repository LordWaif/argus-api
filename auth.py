import argilla as rg
def auth():
    rg.init(
        api_url="http://procyon.tce.pi.gov.br:6900",
        api_key="admin.apikey",
        workspace='admin'
    )
    return rg

from functools import wraps
from flask import request,Response

def token_required(secret):
    def wrapper(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            token = None
            if "Authorization" in request.headers:
                token = request.headers["Authorization"]
            if not token:
                return Response(f'Token não está presente', status=401)
            else:
                print(token,secret)
                if token == secret:
                    return f( *args, **kwargs)
                else:
                    return Response(f'Token não autorizado', status=401)
        return decorated
    return wrapper