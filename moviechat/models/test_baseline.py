import dill
dill._dill._reverse_typemap['ClassType'] = type
mymodel = dill.load(open ("baseline","rb"))
answer = mymodel.generate_text(context=str(input("Talk to the chatbot:")))
print(answer)