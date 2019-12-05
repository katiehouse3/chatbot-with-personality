from django.shortcuts import render
import dill

# Create your views here.
def home(request):
    return render(request, 'moviechat/index.html', {})

import os
from django.http import JsonResponse

CURRENT_DIR = os.path.dirname(__file__)
model_file = os.path.join(CURRENT_DIR, 'models/baseline.pkl')
dill._dill._reverse_typemap['ClassType'] = type
dill.settings['recurse'] = True
mymodel = dill.load(open (model_file,"rb"))
# Create your views here.

def api_chat_response(request):
    print('went to response!')
    chat = request.GET['userinput']
    print('got chat', chat)
    result = mymodel.generate_text(context=str(chat))
    return (JsonResponse(result, safe=False))