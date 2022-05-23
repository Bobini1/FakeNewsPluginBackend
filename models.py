from flask_marshmallow import Marshmallow
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
ma = Marshmallow()


class Post(db.Model):
    __tablename__ = 'blogposts'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(120), index=True)
    content = db.Column(db.Text, index=True)
    date = db.Column(db.DateTime, index=True)
    tag = db.Column(db.String(120), index=True)
    cover = db.Column(db.String(120), index=True)

    def __repr__(self):
        return '<Post: %r>' % (self.title)


class PostSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Post
        include_fk = True