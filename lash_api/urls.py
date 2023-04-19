"""lash_api URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path
from . import factorization_view
urlpatterns = [
    path('polls/', include('polls.urls')),
    path('factorization/', factorization_view.index, name='index'),
    path('admin/', admin.site.urls),
]
#python manage.py runserver 127.0.0.1:8080
#http://127.0.0.1:8080/factorization/?buid=username_1373876832005&modelid=2&prevX=35.144731101693&prevY=33.41113910079&prevDeck=0&smas_db_location_bound_meters=0.0001&oids=100,101,122,124
#http://172.104.245.69/factorization/?buid=username_1373876832005&modelid=2&prevX=35.144478604108&prevY=33.411452583969&prevDeck=-1&smas_db_location_bound_meters=0.0001&oids=100,100,103
#http://172.104.245.69/factorization/?prevX=-1&prevY=-1&prevDeck=-1&oids=2,44,47
#http://127.0.0.1:8080/factorization/?prevX=-1&prevY=-1&prevDeck=-1&oids=2,44,47
#http://127.0.0.1:8080/factorization/?prevX=3&oids=2,44,47
#http://127.0.0.1:8080/factorization/?oids=2,44,47
#http://10.16.30.155:8000/polls/?oids=2,44,47
#http://10.16.30.155:8000/factorization/?oids=2,44,47