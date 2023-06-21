from flask import request,Response,g,jsonify
import sqlite3
from app import app
from app import DATABASE
import json
from datetime import datetime
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
        "batch_size" : "5",
        "isMulti_label" : true
    }
    """
    dados = request.get_json()
    try:
        registro = Dataset(
            id=dados['id'],
            name=dados['name'],
            rotulados=int(dados['rotulados']),
            total=int(dados['total']),
            batch_size=int(dados['batch_size']),
            actual_batch=int(dados['actual_batch']),
            isMulti_label = bool(dados['isMulti_label'])
            )
        db.session.add(registro)
        db.session.commit()
    except:
        db.session.rollback()
        dataset = Dataset.query.get(dados['id'])
        setattr(dataset, 'rotulados', 0)
        setattr(dataset, 'actual_batch', 0)
        db.session.commit()
    return Response('Requisição POST recebida com sucesso!', status=200)

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
        #ac = getattr(dataset,k)
        setattr(dataset, k, v)
    db.session.commit()
    return jsonify(dataset.as_dict())

@app.route('/datasets/<string:dataset_id>/metrics',methods=['POST'])
def post_metrics(dataset_id:str):
    """
    BODY:
    {
        "accuracy": float,
        "hamming_loss": float,
        "trusting": float,
        "batch_id": int,
        "jensenshannon": float,
        "precision" float,
        "recall" : float,
        "f1_score" : float

    }
    """
    dataset:Dataset = Dataset.query.get(dataset_id)
    dados = request.get_json()
    if not dataset:
        return jsonify({'mensagem': 'Dataset não encontrado'}), 404
    registro:Metricas = Metricas.query.filter(Metricas.dataset_id == dataset_id).filter(Metricas.batch_id == int(dados['batch_id'])).first()
    if not registro:
        registro = Metricas(batch_id=int(dados['batch_id']),dataset_id=dataset_id)
    dados = request.get_json()
    for k,v in dados.items():
        setattr(registro, k, v)
    dataset.metricas.append(registro)
    db.session.add(registro)
    db.session.commit()
    return Response('Requisição POST recebida com sucesso!', status=200)

@app.route('/datasets/<string:dataset_id>/metrics',methods=['GET'])
def list_metrics(dataset_id:str):
    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        return jsonify({'mensagem': 'Dataset não encontrado'}), 404
    metricas = [metrica.as_dict() for metrica in dataset.metricas]
    return jsonify(metricas)

@app.route('/datasets/<string:dataset_id>/metrics',methods=['DELETE'])
def delete_metrics(dataset_id:str):
    dataset = Dataset.query.get_or_404(dataset_id)
    for metric in dataset.metrics:
        db.session.delete(metric)
    db.session.commit()
    return 'As métricas associadas ao conjunto de dados foram excluídas com sucesso.'

@app.route('/datasets/<string:dataset_id>/status',methods=['POST'])
def post_status(dataset_id:str):
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
def list_status(dataset_id:str):
    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        return jsonify({'mensagem': 'Dataset não encontrado'}), 404
    status = [_status.as_dict() for _status in dataset.status]
    return jsonify(status)

@app.route('/status/<string:status_id>',methods=['DELETE'])
def del_status(status_id:str):
    status = Status.query.get(status_id)
    _s = jsonify(status.as_dict())
    if not status:
        return jsonify({'mensagem': 'Status não encontrado'}), 404
    db.session.delete(status)
    db.session.commit()
    return _s

@app.route('/datasets/<string:dataset_id>/checkpoint',methods=['POST'])
def post_checkpoint(dataset_id:str):
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
def list_checkpoint(dataset_id:str):
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

@app.route('/datasets/<string:dataset_id>',methods=['DELETE'])
def del_dataset(dataset_id:str):
    dataset:Dataset = Dataset.query.get_or_404(dataset_id)
    bckp = {'dataset':dataset.as_dict(),'metricas':[_.as_dict() for _ in dataset.metricas]}
    if not dataset:
        return jsonify({'mensagem': 'Dataset não encontrado'}), 404
    backup_filename = f'argus-api/backups/{datetime.now()}_backup_resource_{dataset_id}.json'
    with open(backup_filename, 'w') as backup_file:
        backup_file.write(json.dumps(bckp,default=str))
    for _ in dataset.metricas:
        db.session.delete(_)
    db.session.delete(dataset)
    db.session.commit()
    return jsonify({'mensagem': 'Recurso deletado'}), 200

app.run(host='0.0.0.0',port=5001)
