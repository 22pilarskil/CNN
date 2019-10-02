# chat/urls.py
from django.urls import path, re_path

from .views import *

app_name = 'chat'

urlpatterns = [
    path('', index, name='index'),
    re_path(r'^(?P<room_name>[^/]+)/$', room, name='room'),
    #path('<str:room_name>/', room, name='room'),
]
