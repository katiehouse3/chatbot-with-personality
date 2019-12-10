from django.urls import path, re_path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
<<<<<<< HEAD
    re_path(r'^ajax/chat_ngram/$', views.chat_ngram, name='chat_ngram'),
    re_path(r'^ajax/chat_rnn/$', views.chat_rnn, name='chat_rnn'),
    re_path(r'^ajax/write_eval/$', views.write_eval, name='write_eval')
]
=======
    re_path(r'^ajax/api_chat_response/$', views.api_chat_response, name='api_chat_response'),
    re_path(r'^ajax/write_eval/$', views.write_eval, name='write_eval')
]
>>>>>>> bf1e1396fd2742fcecd4d3c3190ea17d1ff5899e
