from flask import Flask, redirect, render_template, session,request,Markup
import numpy as np
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import json
import pickle
import requests
from flask_mail import Mail
from datetime import datetime
import os
import math
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9

# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()
# Loading crop recommendation model

crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


# =========================================================================================

# Custom functions for calculations

def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = params['weather_api_key']
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None



def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction



# ============================================================================================
# Data from config. json is used to configure virtual machine.
local_server = True
with open('config.json', 'r') as c:
    params = json.load(c)["params"]

app = Flask(__name__)
app.config['UPLOAD_FOLDER']  = params['upload_location']
app.secret_key = params['super_secret_key']

# set up SMTP with your Flask application and to send messages from your views and scripts.
app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT='465',
    MAIL_USE_SSL=True,
    MAIL_USERNAME=params['gmail-user'],
    MAIL_PASSWORD=params['gmail-password']
)

mail = Mail(app)

# =================================================================================
# Connection of the database.
# if(local_server):
    # app.config['SQLALCHEMY_DATABASE_URI'] = params['local_uri']
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgres://crhkqutmhblrwx:655b7a48282420a525f55ea6865ff150464b5b64d09d89cce857f78976356ca2@ec2-54-83-21-198.compute-1.amazonaws.com:5432/dcjhvqaheqhuq7'
# else:
    # app.config['SQLALCHEMY_DATABASE_URI'] = params['prod_uri']
db = SQLAlchemy(app)

# Make class for the particular contact  table in database.
class contact(db.Model):

    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    phone = db.Column(db.String(12), nullable=False)
    message = db.Column(db.String(120), nullable=False)
    date = db.Column(db.String(12), nullable=True)
    email = db.Column(db.String(20), nullable=False)

# Make class for the particular post  table in database.
class Post(db.Model):

    sno = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(80), nullable=False)
    tagline = db.Column(db.String(80), nullable=False)
    slug = db.Column(db.String(25), nullable=False)
    content = db.Column(db.Text(), nullable=False)
    date = db.Column(db.String(12), nullable=True)
    img_file = db.Column(db.String(12), nullable=True)

#=========================================Flask App================================================

# =============================================================================================
# Making routing 
@app.route("/")
def home():

    # using jinja for loop for fetch the post from database to show in home page.
    posts = Post.query.filter_by().all()
    last = math.ceil(len(posts)/int(params['no_of_post']))

    #Pagination and check how many post is present.
    page = request.args.get('page')
    if(not str(page).isnumeric()):
        page = 1
    page=int(page)
    # Basically it is doing slicing
    # posts = posts[(page-1)*int(params['no_of_post']):(page-1)*int(params['no_of_post'])+int(params['no_of_post'])]
    j = (page-1)*int(params['no_of_post'])
    posts = posts[j:j+1]
    # Logic of prev and next page
    if(page==1):
            prev="#"
            next="/?page="+ str(page+1)
    elif(page==last):
            prev="/?page="+ str(page-1)
            next="#"
    else:
            prev="/?page="+ str(page-1) 
            next="/?page="+ str(page+1) 

    return render_template('index.html', posts=posts,prev=prev, next=next)

# Routing for post by using slug.
@app.route("/post/<string:post_slug>", methods=['GET'])
def post_route(post_slug):
    post = Post.query.filter_by(slug=post_slug).first()
    return render_template('post.html', post=post)

#  uploader form dashboard page 
@app.route("/uploader", methods=['GET', 'POST'])
def uploader():
    # if Admin is logged in so he can upload a file.
    if('user' in session and session['user'] == params['admin_user']):
        if(request.method == 'POST'):
            f = request.files['file1']
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
            return render_template('dashboard.html')

# Dashboard page
@app.route("/dashboard", methods=['GET', 'POST'])
def dashboard():
# if Admin is logged in so he will redirected to the dashboard. 
    if('user' in session and session['user'] == params['admin_user']):
        posts = Post.query.all()
        return render_template('dashboard.html', posts=posts)
# Admin login data.
    if request.method == 'POST':
        username = request.form.get('uname')
        password = request.form.get('pass')
        if(username == params['admin_user'] and password == params['admin_password']):
            session['user'] = username
            posts = Post.query.all()
            return render_template('dashboard.html', posts=posts)
    else:
        return render_template('login.html')

# Edit page by serial number.
@app.route("/edit/<string:sno>", methods=['GET', 'POST'])
def edit(sno):
    if('user' in session and session['user'] == params['admin_user']):
        if request.method == 'POST':
            box_title = request.form.get('title')
            tline = request.form.get('tline')
            slug = request.form.get('slug')
            content = request.form.get('content')
            img_file = request.form.get('img_file')
            date = datetime.now()
#  Check the condition for add post and edit post.
            if sno=='0':
                post = Post(title=box_title, slug=slug, content=content,
                            tagline=tline, img_file=img_file, date=date)
                db.session.add(post)
                db.session.commit()
            else:
                post = Post.query.filter_by(sno=sno).first()
                post.title = box_title
                post.slug = slug
                post.tagline = tline
                post.content = content
                post.img_file = img_file
                post.date = date
                db.session.commit()
                return redirect('/edit/'+sno) 
    post = Post.query.filter_by(sno=sno).first()
    return render_template('edit.html', post=post, sno=sno)


# Render the about page.
@app.route("/about")
def about():
    return render_template('about.html')

# Render the contact page.
@app.route("/contact", methods=['GET', 'POST'])
def contact_view():
    if(request.method == 'POST'):
#  Add entry to the contact table in database
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        message = request.form.get('message')
        entry = contact(name=name, phone=phone, message=message,date=datetime.now(), email=email)
        db.session.add(entry)
        db.session.commit()
#  Using mail api for sending the message from blog.
        mail.send_message('New message from Blog', sender=email, recipients=[params['gmail-user']],body=name + "\n" + message + "\n" + phone + "\n" + email)
    return render_template('contact.html')

# functioning of delete button in dashboard.
@app.route("/delete/<string:sno>", methods=['GET', 'POST'])
def delete(sno):
    if('user' in session and session['user'] == params['admin_user']):
        post = Post.query.filter_by(sno=sno).first()
        db.session.delete(post)
        db.session.commit()
        return redirect('/dashboard')

# functioning of logout button in dashboard.
@app.route("/logout")
def logout():
    session.pop('user')
    return redirect('/dashboard')

# render crop recommendation form page
@ app.route('/crop-recommend')
def crop_recommend():
    title = "Meadow-The Farmer's Guide - Crop Recommendation"
    return render_template('crop.html', title=title)

# RENDER PREDICTION PAGES
# render crop recommendation result page
@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = "Meadow-The Farmer's Guide - Crop Recommendation"

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        city = request.form.get("city")

        if weather_fetch(city) != None:  
            temperature, humidity = weather_fetch(city) #take out data from api using this function weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]
            return render_template('crop-result.html', prediction=final_prediction, title=title)
        else:
            return render_template('try_again.html', title=title)

# render fertilizer recommendation result page
@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = "Meadow-The Farmer's Guide - Fertilizer Suggestion"
    return render_template('fertilizer.html', title=title)

# RENDER PREDICTION PAGES
# render fertilizer recommendation result pag
@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = "Meadow-The Farmer's Guide - Fertilizer Suggestion"

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])

    df = pd.read_csv('Data/fertilizer.csv') # Read fertilizer.csv file for ideal value of planet

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key])) #take out the result from fertilizer_dic by using key value pairs.

    return render_template('fertilizer-result.html', recommendation=response, title=title)



# render disease prediction result page
@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)

# app.run(debug=True)
app.run()
