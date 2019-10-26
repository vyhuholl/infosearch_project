from django.db import models


class TextCorpora(models.Model):
    text = models.TextField()
    text_lemmatized = models.TextField()

    class Meta:
        db_table = "quora_question_pairs_rus"
        managed = False
