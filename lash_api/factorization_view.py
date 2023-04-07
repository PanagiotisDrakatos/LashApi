from rest_framework.decorators import api_view
from django.http import JsonResponse, HttpResponse
from lash_api.LashFactorization import Factorization
import json
import numpy as np
from collections import OrderedDict
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
@api_view(['GET', 'POST'])
def index(request):
    if request.method == "GET":
        matrix = Factorization()
        buid = request.GET['buid']
        modelid = int(request.GET['modelid'])
        smas_db_location_bound_meters=request.GET['smas_db_location_bound_meters'];
        X=request.GET['prevX']
        Y=request.GET['prevY']
        PREV_DECK = request.GET['prevDeck']
        oids = request.GET['oids']
        oids = oids.split(",")
        for i in range(len(oids)):
            oids[i] = int(oids[i])
        print(oids)
        res = matrix.reccomend(oids,buid,modelid,X,Y,PREV_DECK,smas_db_location_bound_meters)
        json_res=json.dumps(res, separators=(',', ':'), ensure_ascii=False, cls=NpEncoder)
        print(json_res)
       # json_res=json.loads(json_res)
       # print(json_res)
        return HttpResponse(json_res);
