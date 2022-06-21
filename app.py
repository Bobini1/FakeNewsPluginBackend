import sqlalchemy
from flask import Flask, request, jsonify, make_response
from flask_restful import Resource, Api, abort
from marshmallow import ValidationError

from models import db, ma, Article, ArticleSchema, User, UserSchema, UserPreference, UserPreferenceSchema, ArticleRequestSchema
import nlp_analyzer
import sqlite3
from sqlite3 import Error
from datetime import datetime
from flask_cors import CORS
from flask import json
from os.path import dirname

article_schema = ArticleSchema()
article_request_schema = ArticleRequestSchema()
articles_schema = ArticleSchema(many=True)
user_schema = UserSchema()
users_schema = UserSchema(many=True)
user_preferences_schema = UserPreferenceSchema()
user_preferencess_schema = UserPreferenceSchema(many=True)


def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()


app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
api = Api(app)


create_connection('test.db')
db.init_app(app)
ma.init_app(app)


with app.app_context():
    db.create_all()


@app.route("/is_real", methods=['POST'])
def is_real():
    json_data = request.get_json()
    if not json_data:
        return {"message": "No input data provided"}, 400
    try:
        data = article_request_schema.load(json_data)
        if article := Article.query.filter_by(url=data["url"]).first():
            return article.score
        else:
            score = nlp_analyzer.is_real(data["content"])
            article = Article(url=data["url"], score=score, date=data["date"], source_url=dirname(data["url"]),
                              isReviewRequested=False)
            db.session.add(article)
            db.session.commit()
            return score
    except ValidationError as err:
        return jsonify({"message": err.messages}), 422


@app.route("/request_review", methods=['POST'])
def request_review():
    json_data = request.get_json()
    if not json_data:
        return {"message": "No input data provided"}, 400
    try:
        data = article_request_schema.load(json_data)
        if article := Article.query.filter_by(url=data["url"]).first():
            article.isReviewRequested = True
            db.session.commit()
            return "Review requested"
        else:
            return abort(404)
    except ValidationError as err:
        return jsonify({"message": err.messages}), 422


app.run(debug=True)
