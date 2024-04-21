from django.contrib import admin
from .models import FlightInfo, RecoveredBody

# Register your models here.


admin.site.register(FlightInfo)
admin.site.register(RecoveredBody)