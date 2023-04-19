from flask import Flask
app = Flask(__name__)

DATABASE = 'metricas.bd'
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DATABASE}'