from flask import Blueprint, render_template, redirect, url_for, request
from music.pieces.song import *
from web.data.song import riffs, phrases, tracks

tracks_bp = Blueprint('tracks', __name__, template_folder='templates', static_folder='static', url_prefix='/tracks')


@tracks_bp.route('/', methods=['GET'])
def get_tracks():
    return render_template('tracks.html', tracks=tracks)


@tracks_bp.route('/delete/<index>', methods=['POST'])
def delete_track(index):
    tracks.pop(int(index)-1)
    return redirect(url_for('tracks.get_tracks'))


@tracks_bp.route('/edit/<index>', methods=['POST'])
def edit_track(index):
    return redirect(url_for('tracks.get_tracks'))


@tracks_bp.route('/new', methods=['POST'])
def new_track():
    return redirect(url_for('tracks.get_tracks'))
