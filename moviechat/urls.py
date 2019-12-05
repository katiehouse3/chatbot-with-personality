from django.urls import path, re_path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    re_path(r'^ajax/api_chat_response/$', views.api_chat_response, name='api_chat_response')
]