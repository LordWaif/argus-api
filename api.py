from flask import request,Response,g,jsonify
import sqlite3
from app import app
from app import DATABASE
from orm import(
    Metricas,
    Dataset,
    db,
    Status,
    Checkpoint
)

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

@app.route('/datasets',methods=['POST'])
def post_dataset():
    """
    Formato do body
    {
        "name" : "lima",
        "rotulados" : "0",
        "total" : "1000",
        "batch_size" : "5"
    }
    """
    dados = request.get_json()
    try:
        registro = Dataset(
            id=dados['id'],
            name=dados['name'],
            rotulados=int(dados['rotulados']),
            total=int(dados['total']),
            batch_size=int(dados['batch_size'])
            )
        db.session.add(registro)
        db.session.commit()
        return Response('Requisição POST recebida com sucesso!', status=200)
    except:
        return Response('BAD_REQUEST', status=400)

@app.route('/datasets',methods=['GET'])
def get_dataset():
    id = request.args.get('id')
    if id:
        dataset = Dataset.query.filter(Dataset.id==id).first()
        return jsonify(dataset.as_dict())
    else:
        dataset = Dataset.query.all()
        dataset = [i.as_dict() for i in dataset]
        return jsonify(dataset)
    
@app.route('/datasets/<string:dataset_id>',methods=['PATCH'])
def patch_dataset(dataset_id):
    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        return jsonify({'mensagem': 'Dataset não encontrado'}), 404
    updates = request.get_json()
    for k,v in updates.items():
        ac = getattr(dataset,k)
        setattr(dataset, k, ac+v)
    db.session.commit()
    return jsonify(dataset.as_dict())

@app.route('/datasets/<string:dataset_id>/metrics',methods=['POST'])
def post_metrics(dataset_id):
    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        return jsonify({'mensagem': 'Dataset não encontrado'}), 404
    """
    Formato do body
    {
        "accuracy": 0.86,
        "hamming_loss": 0.25,
        "trusting": 0.89,
        "batch_id": 1,
    }
    """
    dados = request.get_json()
    registro = Metricas(
        accuracy = float(dados['accuracy']),
        hamming_loss = float(dados['hamming_loss']),
        trusting = float(dados['trusting']),
        batch_id=int(dados['batch_id']),
        dataset_id=dataset_id)
    dataset.metricas.append(registro)
    db.session.add(registro)
    db.session.commit()
    return Response('Requisição POST recebida com sucesso!', status=200)

@app.route('/datasets/<string:dataset_id>/metrics',methods=['GET'])
def list_metrics(dataset_id):
    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        return jsonify({'mensagem': 'Dataset não encontrado'}), 404
    metricas = [metrica.as_dict() for metrica in dataset.metricas]
    return jsonify(metricas)

@app.route('/datasets/<string:dataset_id>/status',methods=['POST'])
def post_status(dataset_id):
    """
    Formato do body
    {
        "name" : "checkpoint",
        "batch_id" : "0",
    }
    """
    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        return jsonify({'mensagem': 'Dataset não encontrado'}), 404
    
    dados = request.get_json()
    registro = Status(
        name = dados['name'],
        batch_id=dados['batch_id'],
        dataset_id=dataset_id)
    dataset.status.append(registro)
    db.session.add(registro)
    db.session.commit()
    return Response('Requisição POST recebida com sucesso!', status=200)

@app.route('/datasets/<string:dataset_id>/status',methods=['GET'])
def list_status(dataset_id):
    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        return jsonify({'mensagem': 'Dataset não encontrado'}), 404
    status = [_status.as_dict() for _status in dataset.status]
    return jsonify(status)

@app.route('/datasets/<string:dataset_id>/checkpoint',methods=['POST'])
def post_checkpoint(dataset_id):
    """
    Formato do body
    {
        "batch_id" : "0",
        "auto_labeling" : "false"
    }
    """
    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        return jsonify({'mensagem': 'Dataset não encontrado'}), 404
    
    dados = request.get_json()
    registro = Checkpoint(
        batch_id=dados['batch_id'],
        dataset_id=dataset_id,
        isAutoLabeling=bool(dados['auto_labeling']))
    dataset.checkpoints.append(registro)
    db.session.add(registro)
    db.session.commit()
    return Response('Requisição POST recebida com sucesso!', status=200)

@app.route('/datasets/<string:dataset_id>/checkpoint',methods=['GET'])
def list_checkpoint(dataset_id):
    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        return jsonify({'mensagem': 'Dataset não encontrado'}), 404
    checkpoint = [checkpoint.as_dict() for checkpoint in dataset.checkpoints]
    return jsonify(checkpoint)

@app.route('/')
def homepage():
    return "Bem vindo"

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

app.run(host='0.0.0.0',port=5001)
