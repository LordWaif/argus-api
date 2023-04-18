from flask import Flask,request,Response,g,jsonify
import sqlite3
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from auth import token_required

app = Flask(__name__)
DATABASE = 'metricas.bd'
KEYS = ['accuracy','hamming_loss','trusting','batch']
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DATABASE}'
db = SQLAlchemy(app)
from decouple import config
SECRET_KEY = config('SECRET_KEY') or 'this is a secret'
app.config['SECRET_KEY'] = SECRET_KEY

class DictMixin(object):
    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class Metricas(db.Model,DictMixin):
    id = db.Column(db.Integer, db.Sequence('stats_id_seq', increment=1), primary_key=True)
    accuracy = db.Column(db.Float)
    hamming_loss = db.Column(db.Float)
    trusting = db.Column(db.Float)
    batch = db.Column(db.Integer)
    data_hora = db.Column(db.DateTime, default=datetime.utcnow)

class Status(db.Model,DictMixin):
    id = db.Column(db.Integer, db.Sequence('stats_id_seq', increment=1), primary_key=True)
    status= db.Column(db.String)
    data_hora = db.Column(db.DateTime, default=datetime.utcnow)

class Dataset(db.Model,DictMixin):
    __tablename__ = "dataset"
    id = db.Column(db.Integer, db.Sequence('stats_id_seq', increment=1), primary_key=True)
    dataset_name = db.Column(db.String, unique=True)
    checkpoints = db.relationship('Checkpoint', backref='dataset', lazy='dynamic')

class Checkpoint(db.Model,DictMixin):
    __tablename__ = 'checkpoint'
    id = db.Column(db.Integer, db.Sequence('stats_id_seq', increment=1), primary_key=True)
    data_hora = db.Column(db.DateTime, default=datetime.utcnow)
    batch = db.Column(db.Integer)
    dataset_id = db.Column(db.BigInteger, db.ForeignKey('dataset.id'), nullable=False)


with app.app_context():
    db.create_all()

def conectar_bd():
    conexao = sqlite3.connect(DATABASE)
    conexao.row_factory = sqlite3.Row
    return conexao

def get_bd():
    if 'bd' not in g:
        g.bd = conectar_bd()
    return g.bd

@app.teardown_appcontext
def fechar_bd(erro):
    if hasattr(g, 'bd'):
        g.bd.close()

def isValid(json):
    if all(chave in json for chave in KEYS):
        return True
    else:
        return False

@app.route('/')
def homepage():
    return "Bem vindo"

@app.route('/metrics',methods=['POST'])
def post_metrics():
    """
    Formato do body
    {
        "accuracy": 0.86,
        "hamming_loss": 0.25,
        "trusting": 0.89,
        "batch": 1
    }
    """
    dados = request.get_json()
    if(isValid(dados)):
        registro = Metricas(accuracy = dados['accuracy'],hamming_loss = dados['hamming_loss'],trusting = dados['trusting'],batch=dados['batch'])
        db.session.add(registro)
        db.session.commit()
        return Response('Requisição POST recebida com sucesso!', status=200)
    else:
        return Response(f'Keywords do body não estão no formato correto keywords: {list(dados.keys())}\nEsperadas: {KEYS}', status=400)
    
@app.route('/metrics',methods=['GET'])
def get_metrics():
    metrica = Metricas.query.order_by(Metricas.data_hora.desc()).first()
    if metrica == None:
        metrica = {'accuracy':"0.0",'hamming_loss':"1",'trusting':"0",'batch':"-1"}
    else:
        metrica = metrica.as_dict()
    return jsonify(metrica)

@app.route('/checkpoint',methods=['POST'])
def post_checkpoint():
    """
    Formato do body
    {
        "batch": 1
    }
    """
    dados = request.get_json()
    try:
        registro = Checkpoint(batch=dados['batch'])
        db.session.add(registro)
        db.session.commit()
        return Response('Requisição POST recebida com sucesso!', status=200)
    except:
        return Response('BAD_REQUEST', status=400)
    
@app.route('/checkpoint/<dataset_name>',methods=['GET'])
def get_checkpoint(dataset_name):
    checkpointer = Checkpoint.query.filter_by(dataset_name=dataset_name).query.order_by(Checkpoint.data_hora.desc()).first()
    if checkpointer != None:
        checkpointer = checkpointer.as_dict()
    return jsonify(checkpointer)

@app.route('/dataset',methods=['POST'])
def post_dataset():
    """
    Formato do body
    {
        "dataset_name": "lima"
    }
    """
    dados = request.get_json()
    try:
        registro = Dataset(dataset_name=dados['dataset_name'])
        db.session.add(registro)
        db.session.commit()
        return Response('Requisição POST recebida com sucesso!', status=200)
    except:
        return Response('BAD_REQUEST', status=400)
    
@app.route('/dataset/<dataset_name>',methods=['GET'])
def get_dataset(dataset_name):
    dataset = Dataset.query.filter_by(dataset_name=dataset_name).first()
    return dataset

@app.route('/status',methods=['GET'])
def get_status():
    status = Status.query.order_by(Status.data_hora.desc()).first()
    if status == None:
        status = {'status':'Desativado'}
    else:
        status = status.as_dict()
    return jsonify(status)

@app.route('/status',methods=['POST'])
def post_status():
    """
    Formato do body
    {
        "status": "checkpoint"
    }
    """
    dados = request.get_json()
    registro = Status(status = dados['status'])
    db.session.add(registro)
    db.session.commit()
    return Response('Requisição POST recebida com sucesso!', status=200)

app.run(host='0.0.0.0',port=5001)
