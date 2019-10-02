# chat/routing.py
from django.urls import path, re_path

from .consumers import ChatConsumer

websocket_urlpatterns = [
    #path('ws/chat/<str:room_name>/$', ChatConsumer),
    re_path(r'^ws/chat/(?P<room_name>[^/]+)/$', ChatConsumer),
]
