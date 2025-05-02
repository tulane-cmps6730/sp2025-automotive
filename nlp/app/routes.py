from flask import render_template, flash, redirect, session
from . import app
from .forms import MyForm
from .. import clf_path
from .. import nlp

import pickle
import sys


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    answer = None
    form = MyForm()
    if form.validate_on_submit():
        question = form.input_field.data
        answer = str(nlp.web_chat(question))
        return render_template('myform.html', title = '', form=form, answer=answer)
    return render_template('myform.html', title = '', form=form, answer=None)

