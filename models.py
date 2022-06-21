from flask_marshmallow import Marshmallow
from marshmallow import fields
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
ma = Marshmallow()


class Article(db.Model):
    __tablename__ = 'article'

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, index=True)
    url = db.Column(db.Text, index=True, unique=True)
    source_url = db.Column(db.Text, index=True)
    score = db.Column(db.Integer)
    isReviewRequested = db.Column(db.Boolean)


class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    email = db.Column(db.String(120), index=True, unique=True)
    preference = db.relationship('UserPreference')


class UserPreference(db.Model):
    __tablename__ = 'user_preference'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))


class ArticleSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Article
        dateformat = '%Y-%m-%dT%H:%M:%S'
        load_instance = True


class UserPreferenceSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = UserPreference
        load_instance = True


class UserSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = User
        load_instance = True
    user_preferences = fields.Nested(UserPreferenceSchema)


class ArticleRequestSchema(ma.Schema):
    content = fields.String(required=True)
    url = fields.String(required=True)
    date = fields.DateTime(required=True)
