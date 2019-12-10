from django.urls import path, re_path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    re_path(r'^ajax/chat_ngram/$', views.chat_ngram, name='chat_ngram'),
    re_path(r'^ajax/chat_rnn/$', views.chat_rnn, name='chat_rnn'),
    re_path(r'^ajax/write_eval/$', views.write_eval, name='write_eval')
]
