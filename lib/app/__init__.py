"""
API Server for onregistry web services
"""
import importlib
import json
import logging
from os import environ

from flask import Flask, jsonify, request, redirect, Response
from pymongo import MongoClient
from redis import StrictRedis
from rq import Queue
import rq_dashboard

from lib.conf import rq_dashboard_settings
from lib.controllers.rq_dashboard_controller import rq_dashboard_blueprint

APP = Flask(__name__)
MONGO = MongoClient(environ['MONGO_PORT_27017_TCP_ADDR'], 27017)
DB = MONGO.database['dist_sys']
REDIS = StrictRedis(host='redis')
Q = Queue(connection=REDIS)

APP.config.from_object(rq_dashboard_settings)
APP.register_blueprint(rq_dashboard_blueprint,
                        static_folder='assets/static',
                        url_prefix='/rq')

@APP.route('/')
def api_live():
    """ root endpoint; serves to notify that the application is live """
    return Response(json.dumps({'message' : 'Api is live.'}), 200)

@APP.route('/logging_example')
def logging_example():
    APP.logger.info(environ)
    APP.logger.error(vars())
    APP.logger.warning('this is a warning, move along.')
    APP.logger.debug('this is a debug statement.')
    APP.logger.critical('stop everything!')
    return Response(json.dumps({'message' : 'check the logs!'}), 200)
