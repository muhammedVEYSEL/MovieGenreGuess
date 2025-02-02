from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('guess.urls')),  # guess uygulamasını ana url'e bağla
]
