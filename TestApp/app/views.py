from flask import render_template
from app import app
@app.route('/')
def home():
    return "hello worldc!"

@app.route('/template')
def template():
   return render_template('home.html')

@app.route('/template2')
def template2():
   return render_template('home2.html')

