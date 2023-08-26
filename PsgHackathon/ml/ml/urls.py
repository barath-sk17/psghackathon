from django.contrib import admin
from django.urls import path
from jobsearch import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.home,name="home"),
    path('createfile',views.createfile,name="createfile"),
    path('transition',views.transition,name="transition"),
    path('freelancer',views.freelancer,name="freelancer"),
    path('jobassign',views.jobassign,name="jobassign"),
    
]
urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
