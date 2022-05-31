import sqlalchemy
from flask import Flask, request, jsonify, make_response
from flask_restful import Resource, Api
from marshmallow import ValidationError

from models import db, ma, Article, ArticleSchema, Source, SourceSchema, User, UserSchema, UserPreference, UserPreferenceSchema
import nlp_analyzer
import sqlite3
from sqlite3 import Error
from datetime import datetime
from flask_cors import CORS
from flask import json

article_schema = ArticleSchema()
articles_schema = ArticleSchema(many=True)
source_schema = SourceSchema()
sources_schema = SourceSchema(many=True)
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
        data = article_schema.load(json_data, session=db.session)
    except ValidationError as err:
        print(err)
        return err.messages, 422
    try:
        db.session.add(data)
        db.session.commit()
    except sqlalchemy.exc.IntegrityError:
        pass
    return json.dumps(nlp_analyzer.is_real(data.content)), 200


#  simplified version for presentation
@app.route("/is_fake", methods=['POST'])
def is_fake():
    json_data = request.get_json()
    return json.dumps(not nlp_analyzer.is_real(json_data["content"])), 200




app.run(debug=True)
