from django import forms
class CreateForm(forms.Form):
    resume_file = forms.FileField(widget=forms.FileInput(attrs={'style':'width: 50%; padding: 12px 20px; margin: 8px 0; display: inline-block; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;'}))
