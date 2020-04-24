from django.db import models
from django.contrib.auth.models import User, Group

# Create your models here.


class Player(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    mobile = models.CharField(max_length=100)
    email = models.EmailField()
    company = models.CharField(max_length=199)
    address = models.CharField(max_length=100)
    image = models.ImageField(upload_to='employer')


    def __str__(self):
        return self.name