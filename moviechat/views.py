import os
from django.http import JsonResponse
from django.shortcuts import render
import dill
from moviechat.models import UserEval
import json
import subprocess
from subprocess import check_output

# Create your views here.


def home(request):
    return render(request, 'moviechat/index.html', {})


dill._dill._reverse_typemap['ClassType'] = type
CURRENT_DIR = os.path.dirname(__file__)
model_file = os.path.join(CURRENT_DIR, 'nlpmodels/baseline.pkl')
baseline = dill.load(open(model_file, "rb"))


def chat_ngram(request):
    print('went to response!')
    chat = request.GET['userinput']
    print('got chat', chat)
    result = baseline.generate_text(context=str(chat))
    return (JsonResponse(result, safe=False))


def chat_rnn(request):
    print('went to response!')
    chat = request.GET['userinput']
    print('got chat', chat)
    proc = subprocess.Popen(["python", os.path.join(
        CURRENT_DIR, 'nlpmodels/RNNChatEval.py'), chat], stdout=subprocess.PIPE)
    text = proc.communicate()[0].decode('utf-8')
    return (JsonResponse(text, safe=False))

# Create your views here.


def home(request):
    return render(request, 'moviechat/index.html', {})


# load the python models
CURRENT_DIR = os.path.dirname(__file__)
model_file = os.path.join(CURRENT_DIR, 'models/baseline.pkl')
dill._dill._reverse_typemap['ClassType'] = type
dill.settings['recurse'] = True
mymodel = dill.load(open(model_file, "rb"))


def api_chat_response(request):
    print('went to response!')
    chat = request.GET['userinput']
    print('got chat', chat)
    result = mymodel.generate_text(context=str(chat))
    return (JsonResponse(result, safe=False))

# write evaluation


def write_eval(request):
    usereval = json.loads(request.GET['usereval'])
    print(usereval)
    UserEval.objects.create(first_name=usereval[0]["firstname"],
                            last_name=usereval[1]["lastname"],
                            genre=usereval[2]["genre"],
                            model=usereval[3]["model"],
                            syntactic_score=usereval[4]["syntactic_score"],
                            semantic_score=usereval[5]["semantic_score"],
                            fun_score=usereval[6]["fun_score"],
                            genre_score=usereval[7]["genre_score"])
    return (JsonResponse('Thanks for evaluating! &#128079;', safe=False))
