from django.db import models

# Create your models here.

class CreateFile(models.Model):
    resume = models.FileField(max_length=25)
