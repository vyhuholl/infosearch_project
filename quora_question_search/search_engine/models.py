from django.db import models


class TextCorpora(models.Model):
    text = models.TextField()
    text_lemmatized = models.TextField()

    class Meta:
        db_table = "text_corpora"
        managed = False
