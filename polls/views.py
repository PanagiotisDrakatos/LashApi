from rest_framework.decorators import api_view
from lash_api.LashML import TensorflowML
from django.http import JsonResponse
import json

@api_view(['GET', 'POST'])
def index(request):
    if request.method == "GET":
        ml_lib=TensorflowML()
        oids = request.GET['oids']
        oids=oids.split(",")
        for i in range(len(oids)):
            oids[i]=int(oids[i])
        print(oids)
        results=ml_lib.predict(oids)
        return JsonResponse(results,safe=False)
