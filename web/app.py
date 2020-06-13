from flask import Flask
from flask import render_template, url_for
from flask_bootstrap import Bootstrap
from livereload import Server
from music.pieces.song import *

app = Flask(__name__, template_folder='templates', static_folder='static')
Bootstrap(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/riffs/<riff_type>')
def riffs(riff_type):
    if riff_type == 'all':
        return render_template('riffs/all_riffs.html', riffs=riffs)
    else:
        return render_template('riffs/' + riff_type + '.html', riffs=riffs[riff_type])


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

    # server = Server(app.wsgi_app)
    # server.watch('**/*.*')
    # server.serve()

    app.run()
