from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("essayscoring",views.score_essay.as_view(),name="essayscoring")
]