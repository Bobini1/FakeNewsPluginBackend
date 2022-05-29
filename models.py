from flask_marshmallow import Marshmallow
from marshmallow import fields
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
ma = Marshmallow()


class Source(db.Model):
    __tablename__ = 'source'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False, unique=True)
    url = db.Column(db.String(120), nullable=False, unique=True)
    reliability = db.Column(db.Float, nullable=False)


class Article(db.Model):
    __tablename__ = 'article'

    id = db.Column(db.Integer, primary_key=True)
    type_of_article = db.Column(db.String(120), index=True)
    content = db.Column(db.Text, index=True, unique=True)
    date = db.Column(db.DateTime, index=True)
    topic = db.Column(db.String(120), index=True)
    country = db.Column(db.String(120), index=True)
    source_id = db.Column(db.Integer, db.ForeignKey('source.id'))
    source = db.relationship("Source")


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


class SourceSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Source
        load_instance = True


class ArticleSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Article
        dateformat = '%Y-%m-%dT%H:%M:%S'
        load_instance = True
    source = fields.Nested(SourceSchema)


class UserPreferenceSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = UserPreference
        load_instance = True


class UserSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = User
        load_instance = True
    user_preferences = fields.Nested(UserPreferenceSchema)