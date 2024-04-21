from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class FlightInfo(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, default=1)
    context = models.TextField(null=True)

    def __str__(self):
        return f"{self.user} - {self.context}"
    
class RecoveredBody(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, default=1)
    crashTime = models.DateTimeField()
    recoveryTime = models.DateTimeField()
    latitude = models.FloatField()
    longitude = models.FloatField()

    def __str__(self):
        return f"{self.latitude}, {self.longitude} - {self.recoveryTime}"