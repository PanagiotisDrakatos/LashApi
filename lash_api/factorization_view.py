from rest_framework.decorators import api_view
from django.http import JsonResponse
from lash_api.LashFactorization import Factorization
import json
import numpy as np

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
        oids = request.GET['oids']
        oids = oids.split(",")
        for i in range(len(oids)):
            oids[i] = int(oids[i])
        print(oids)
        res = matrix.reccomend(oids)
        json_res=json.dumps(res, sort_keys=True, separators=(',', ': '), ensure_ascii=False,cls=NpEncoder)
        json_res=json.loads(json_res)
        return JsonResponse(json_res, safe=False)
