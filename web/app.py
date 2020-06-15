from flask import Flask
from flask import render_template, url_for, request, redirect
from flask_bootstrap import Bootstrap
from livereload import Server
from web.views.riffs_bp import riffs_bp
from web.views.phrases_bp import phrases_bp
from music.pieces.song import *

app = Flask(__name__, template_folder='templates', static_folder='static')
app.register_blueprint(riffs_bp, url_prefix='/riffs')
app.register_blueprint(phrases_bp, url_prefix='/phrases')
Bootstrap(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/tracks')
def tracks():
    return render_template('tracks.html', tracks=tracks)


if __name__ == '__main__':
    json_path = 'D:/PycharmProjects/RiffGAN/data/pieces/songs/json/test_song.json'

    song = create_song_drom_json(json_path)
    riffs = song.get_all_riffs()
    phrases = song.get_all_phrases()
    tracks = song.get_all_tracks()

    # server = Server(app.wsgi_app)
    # server.watch('**/*.*')
    # server.serve()

    app.run()
