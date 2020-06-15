from flask import Blueprint, render_template, redirect, url_for, request
from music.pieces.song import *
from web.data.song import riffs, phrases, tracks

tracks_bp = Blueprint('tracks', __name__, template_folder='templates', static_folder='static', url_prefix='/tracks')


@tracks_bp.route('/', methods=['GET'])
def get_tracks():
    return render_template('tracks.html', tracks=tracks)
