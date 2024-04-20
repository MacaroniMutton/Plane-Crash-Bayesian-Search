from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class FlightInfo(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, default=1)
    context = models.TextField(null=True)

    def __str__(self):
        return f"{self.user} - {self.context}"