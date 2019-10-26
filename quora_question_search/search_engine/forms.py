from django import forms


class SearchForm(forms.Form):
    query = forms.CharFiled(label="query", max_length=200)
