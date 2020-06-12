from flask import Flask
from flask import render_template
from flask_bootstrap import Bootstrap
from livereload import Server
from music.pieces.song import *

app = Flask(__name__, template_folder='templates')
Bootstrap(app)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/riffs')
def riffs():
    return render_template('riffs.html', riffs=riffs)


@app.route('/phrases')
def phrases():
    return render_template('phrases.html', phrases=phrases)


@app.route('/tracks')
def tracks():
    return render_template('tracks.html', tracks=tracks)


if __name__ == '__main__':
    json_path = 'D:/PycharmProjects/RiffGAN/data/pieces/songs/json/test_song.json'

    song = create_song_drom_json(json_path)
    tracks = song.get_all_tracks()
    phrases = song.get_all_phrases()
    riffs = song.get_all_riffs()

    server = Server(app.wsgi_app)
    server.watch('**/*.*')
    server.serve()
    # app.run()
