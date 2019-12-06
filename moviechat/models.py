from django.conf import settings
from django.db import models
from django.utils import timezone


class UserEval(models.Model):
    #author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    MODELS = (
        ('A', 'ModelA'),
        ('B', 'ModelB'),
        ('C', 'ModelC'),
    )

    GENRES = (
        ('comedy', 'comedy'),
        ('sport', 'Sport'),
        ('biography', 'Biography'),
        ('romance', 'Romance'),
        ('action', 'Action'),
        ('adventure', 'Adventure'),
        ('drama', 'Drama'),
        ('sci-fi', 'Sci-fi'),
        ('family', 'Family'),
        ('fantasy', 'Fantasy'),
        ('musical', 'Musical'),
        ('crime', 'Crime'),
        ('thriller', 'Thriller'),
        ('short', 'Short'),
        ('western', 'Western'),
        ('documentary', 'Documentary'),
        ('horror', 'Horror'),
        ('animation', 'Animation'),
        ('film-noir', 'Film-noir'),
        ('music', 'Music'),
        ('war', 'War'),
        ('mystery', 'Mystery')
    )

    # User first and last name
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)

    # User scores for the testing
    semantic_score = models.IntegerField()
    syntactic_score = models.IntegerField()
    fun_score = models.IntegerField()
    genre_score = models.IntegerField()

    # Which model and genrue the user is evaluating
    model = models.CharField(max_length=1, choices=MODELS)
    genre =  models.CharField(max_length=100, choices=GENRES)

    # Keep track of when the user evaluates the model
    created_date = models.DateTimeField(default=timezone.now)