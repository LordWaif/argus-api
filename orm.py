from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from app import app

db = SQLAlchemy(app)

class DictMixin(object):
    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class Metricas(db.Model,DictMixin):
    __tablename__ = "metricas"
    id = db.Column(db.Integer, db.Sequence('stats_id_seq', increment=1), primary_key=True)

    batch_id = db.Column(db.Integer)
    data_hora = db.Column(db.DateTime, default=datetime.utcnow)

    accuracy = db.Column(db.Float)
    hamming_loss = db.Column(db.Float)
    trusting = db.Column(db.Float)
    jensenshannon = db.Column(db.Float)
    entropy = db.Column(db.Float)

    dataset_id = db.Column(db.BigInteger, db.ForeignKey('dataset.id'), nullable=False)

class Status(db.Model,DictMixin):
    __tablename__ = "status"
    id = db.Column(db.Integer, db.Sequence('stats_id_seq', increment=1), primary_key=True)

    name = db.Column(db.String)
    batch_id = db.Column(db.Integer)
    data_hora = db.Column(db.DateTime, default=datetime.utcnow)

    dataset_id = db.Column(db.BigInteger, db.ForeignKey('dataset.id'), nullable=False)

class Dataset(db.Model,DictMixin):
    __tablename__ = "dataset"
    id = db.Column(db.String, primary_key=True)

    name = db.Column(db.String,unique=True)
    rotulados = db.Column(db.Integer)
    total = db.Column(db.Integer)
    batch_size = db.Column(db.Integer)
    actual_batch = db.Column(db.Integer,nullable=True)

    status = db.relationship('Status', backref='status', lazy='dynamic')
    metricas = db.relationship('Metricas', backref='metricas', lazy='dynamic')
    checkpoints = db.relationship('Checkpoint', backref='checkpoint', lazy='dynamic')

class Checkpoint(db.Model,DictMixin):
    __tablename__ = 'checkpoint'
    id = db.Column(db.Integer, db.Sequence('stats_id_seq', increment=1), primary_key=True)

    batch_id = db.Column(db.Integer)
    isAutoLabeling = db.Column(db.Boolean)
    data_hora = db.Column(db.DateTime, default=datetime.utcnow)

    dataset_id = db.Column(db.BigInteger, db.ForeignKey('dataset.id'), nullable=False)

with app.app_context():
    db.create_all()

    
