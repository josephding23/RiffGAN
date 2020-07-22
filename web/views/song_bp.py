from flask import Blueprint, render_template, redirect, url_for, request
from music.pieces.track.toolkit import *
from music.pieces.track.track import *
from music.pieces.song.toolkit import *
from music.pieces.song.song import *
import pygame

from web.database.song import *

song_bp = Blueprint('song', __name__, template_folder='templates', static_folder='static', url_prefix='/song')

freq = 44100
bitsize = -16
channels = 2
buffer = 1024
pygame.mixer.init(freq, bitsize, channels, buffer)
pygame.mixer.music.set_volume(1)


@song_bp.route('/new_song', methods=['POST'])
def new_song():
    if request.method == 'POST':
        create_new_song_and_duplicate_as_temp()

        return redirect(url_for('song.get_song'))


@song_bp.route('/save_song', methods=['POST'])
def save_song():
    if request.method == 'POST':
        name = request.form['save_name_input']

        save_temp_song_as(name)

        return redirect(url_for('song.get_song'))


@song_bp.route('/', methods=['GET'])
def get_song():
    return render_template('song.html', song=get_temp_song(), genres_list=get_all_genres())


@song_bp.route('/edit', methods=['POST'])
def edit_song():
    if request.method == 'POST':

        title = request.form['edit_title_input']
        songwriter = request.form['edit_songwriter_input']
        genre = request.form['edit_genre_input']

        song = get_temp_song()
        song['title'] = title
        song['songwriter'] = songwriter
        song['genre'] = genre

        save_temp_song(song)

        return redirect(url_for('song.get_song'))


@song_bp.route('/play', methods=['POST'])
def play_song():
    if request.method == 'POST':
        song_info = get_temp_song()
        tracks = get_temp_tracks()

        checked_info = request.form.getlist('include_track')
        checked_list = [int(checked_index)-1 for checked_index in checked_info]

        refresh_included_tracks_info(song_info, tracks, checked_list)

        save_temp_song(song_info)

        song = parse_song_json(song_info)

        song.add_tracks_to_pm()
        song.save_midi()
        song.play_with_no_init()

        return redirect(url_for('song.get_song'))


@song_bp.route('/stop', methods=['POST'])
def stop_song():
    if request.method == 'POST':

        if pygame.mixer.music.get_busy():
            pygame.mixer.music.fadeout(1000)
            pygame.mixer.music.stop()

        return redirect(url_for('song.get_song'))
