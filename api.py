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
@token_required(app.config['SECRET_KEY'])
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
        return Response('Requisição GET recebida com sucesso!', status=200)
    else:
        return Response(f'Keywords do body não estão no formato correto keywords: {list(dados.keys())}\nEsperadas: {KEYS}', status=400)
    
@app.route('/metrics',methods=['GET'])
def get_metrics():
    metrica = Metricas.query.order_by(Metricas.data_hora.desc()).first()
    metrica = metrica.as_dict()
    return jsonify(metrica)

app.run(host='0.0.0.0',port=5001)
