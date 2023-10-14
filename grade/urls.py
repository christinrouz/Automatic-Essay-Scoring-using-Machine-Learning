from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("AES/", include("AES.urls")),
    path("admin/", admin.site.urls),
    
]