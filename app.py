from flask import Flask, request, jsonify, make_response
from flask_restful import Resource, Api
from models import db, ma, Post, PostSchema
import sqlite3
from sqlite3 import Error
from datetime import datetime


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
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
api = Api(app)



create_connection('test.db')
db.init_app(app)
ma.init_app(app)


with app.app_context():
    db.create_all()
    post = Post(title='Hello', content='World', date=datetime.now(), tag='Hello', cover='World')
    db.session.add(post)
    db.session.commit()


class PostController(Resource):  # put application's code here
    posts_schema = PostSchema(many=True)
    post_schema = PostSchema()

    def get(self):
        return PostController.posts_schema.dump(db.session.query(Post).all())


api.add_resource(PostController, '/')

app.run(debug=True)
