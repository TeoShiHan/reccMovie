#app.py
from flask import Flask, render_template, request, jsonify, json
from sqlalchemy import null
from wtforms import StringField, Form
from wtforms.validators import DataRequired, Length
from flask_sqlalchemy import SQLAlchemy 
from Recommender.Recommender import Recommender
import get_img as imdb_img
import movieposters as mp

# region: custom obj
# custom classes

class image_link(object):
    def __init__(self, title, imdbID):
        self.title = title
        self.imdb_id = imdbID
        self.link  = imdb_img.get_img_link(self.imdb_id)
        self.site_link = mp.get_imdb_link_from_id(imdbID)
# endregion




reccTools = Recommender()

# region : declaration
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///movieDB.db'
db = SQLAlchemy(app)
# endregion


# region : Tables
class MoviesCF(db.Model):
    __tablename__ = 'CF_movies'
    imdb_id = db.Column(db.String(200), primary_key = True)
    movie_title = db.Column(db.String(200), unique=True, nullable = False)
    
    def as_dict(self):
        return {'movie_title':self.movie_title, 'imdb_id':self.imdb_id}

class MoviesCB(db.Model):
    __tablename__ = 'CB_movies'
    imdb_id = db.Column(db.String(200), primary_key = True)
    movie_title = db.Column(db.String(200), unique=True, nullable = False)
    
    def as_dict(self):
        return {'movie_title':self.movie_title, 'imdb_id':self.imdb_id}
    
    
class UserIDs(db.Model):
    __tablename__ = 'userID'
    user_id = db.Column(db.String(20), primary_key = True)
    
    def as_dict(self):
        return {'user_id':self.user_id}
    
class Genre(db.Model):
    __tablename__ = 'Genre'
    genre = db.Column(db.String(100), primary_key = True)
    
    def as_dict(self):
        return {'genre':self.genre}
# endregion : Tables


# region : Forms layout
class SearchForm(Form): #create form
    movie_name = \
        StringField(
            'movie_name', 
            validators=[
                DataRequired(), 
                Length(max=40)
            ], 
            render_kw={"placeholder":"Enter movie title"}
        )
# endregion


# program
@app.route('/')
def index():
    form =SearchForm(request.form)
    return render_template('index.html', form=form)

@app.route('/itemBased', methods=['GET', 'POST'])
def itemBased():
    if request.method == "POST":
        
        
        movie_title = request.form['movie_name']
        topN = request.form['points']
        
        recc = reccTools.IBCF(movie_title, int(topN))
        
        reccObj = []
        
        if recc != null:
    
            for title in recc:
                imdb_id = get_movie_id(title)
                reccObj.append(image_link(title, imdb_id))
        
        return render_template('itemBased.html', recc = reccObj, fromT = movie_title)
    
    else:
        return render_template('itemBased.html')


@app.route('/genre', methods=['GET', 'POST'])
def genreBased():
    if request.method == "POST":
        genre = request.form['genre']
        topN = request.form['points']
        recc = reccTools.G(genre, int(topN))
        reccObj = []
        if recc != null:
            for title in recc:
                imdb_id = get_movie_id(title)
                reccObj.append(image_link(title, imdb_id))
        return render_template('genreBased.html', recc = reccObj, fromT = genre)
    
    else:
        return render_template('genreBased.html')


@app.route('/keyWord', methods=['GET', 'POST'])
def keyWord():
    if request.method == "POST":

        movie_title = request.form['movie_name']
        topN = request.form['points']
        
        recc = reccTools.KD(movie_title, int(topN))
        
        reccObj = []
        
        if recc != null:
    
            for title in recc:
                imdb_id = get_movie_id(title)
                reccObj.append(image_link(title, imdb_id))
        
        return render_template('keyword_desc.html', recc = reccObj, fromT = movie_title)
    
    else:
        return render_template('keyword_desc.html')


@app.route('/neuralNetwork', methods=['GET', 'POST'])
def neuralBased():
    if request.method == "POST":
        
        
        user_id = request.form['user_id_s']

        
        topN = request.form['points']
        
        reccPre = reccTools.NCF(int(user_id), int(topN))
        
        try:
            recc = reccPre[1]
        except:
            recc = []
            
        try:
            rated = reccPre[0]
        except:
            rated = []
        

        
        reccObj = []
        ratedObj =[]
        
        for title in recc:

            
            imdb_id = get_movie_id(title)
            
            reccObj.append(image_link(title, imdb_id))
        
        for title in rated:
            
            imdb_id = get_movie_id(title)
            
            ratedObj.append(image_link(title, imdb_id))
        
        return render_template('neuralBased.html', recc = reccObj, rated= ratedObj)
    
    else:
        return render_template('neuralBased.html')


@app.route('/SVD', methods=['GET', 'POST'])
def svdBased():
    if request.method == "POST":
        
        
        user_id = request.form['user_id_s']

        topN = request.form['points']
        
        reccPre = reccTools.SVDUCF(int(user_id), int(topN))
        
        try:
            recc = reccPre[1]
        except:
            recc = []
            
        try:
            rated = reccPre[0]
        except:
            rated = []
        
        reccObj = []
        ratedObj =[]
        
        for title in recc:
            imdb_id = get_movie_id(title)
            reccObj.append(image_link(title, imdb_id))
        
        for title in rated:
            
            imdb_id = get_movie_id(title)
            
            ratedObj.append(image_link(title, imdb_id))
        
        return render_template('svd.html', recc = reccObj, rated= ratedObj)
    
    else:
        return render_template('svd.html')


@app.route('/wr', methods=['GET', 'POST'])
def wrBased():
    if request.method == "POST":
        topN = request.form['points']
        recc = reccTools.WR(int(topN))
        reccObj = []
        if recc != null:
            for title in recc:
                imdb_id = get_movie_id(title)
                reccObj.append(image_link(title, imdb_id))
        return render_template('wrBased.html', recc = reccObj, fromT = "top " + topN + " movies")
    
    else:
        return render_template('wrBased.html')


@app.route('/get_cb_movies')
def cb_movies_dict():
    movies = MoviesCB.query.all()
    dict_list = [movie.as_dict() for movie in movies]
    return jsonify(dict_list)


@app.route('/get_cf_movies')
def cf_movies_dict():
    movies = MoviesCF.query.all()
    dict_list = [movie.as_dict() for movie in movies]
    return jsonify(dict_list)


@app.route('/get_user_id')
def user_dict():
    user_ids = UserIDs.query.all()
    dict_list = [user_id.as_dict() for user_id in user_ids]
    return jsonify(dict_list)

@app.route('/get_genre')
def genre_dict():
    genre = Genre.query.all()
    dict_list = [g.as_dict() for g in genre]
    return jsonify(dict_list)



def get_movie_id(title):
    import pandas as pd
    title_id = pd.read_csv('Dataset/all_movies.csv')
    result = title_id[
                title_id['movie_title'] == title
             ]
    try:
        return pd.Series(result['imdb_id']).to_list()[0]
    except:
        return "tt0113497"


if __name__ == "__main__":
    app.run(debug=True)         # debugging mode , flask app can run
