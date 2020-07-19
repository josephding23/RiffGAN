from flask import Flask
from flask import render_template, url_for, request, redirect, current_app, send_from_directory
from flask_bootstrap import Bootstrap
from livereload import Server
from web.views.riffs_bp import riffs_bp
from web.views.phrases_bp import phrases_bp
from web.views.tracks_bp import tracks_bp
from web.views.song_bp import song_bp
from web.database.song import *

app = Flask(__name__, template_folder='templates', static_folder='static')
app.register_blueprint(riffs_bp, url_prefix='/riffs')
app.register_blueprint(phrases_bp, url_prefix='/phrases')
app.register_blueprint(tracks_bp, url_prefix='/tracks')
app.register_blueprint(song_bp, url_prefix='/song')
Bootstrap(app)


@app.route('/')
def index():
    existed_songs = get_all_existed_songs()
    return render_template('index.html', existed_songs=existed_songs)


@app.route('/open_song/<song_name>', methods=['POST'])
def open_song(song_name):
    if request.method == 'POST':
        load_song_and_duplicate_as_temp(song_name)

        return redirect(url_for('song.get_song'))


@app.route('/delete_song/<song_name>', methods=['POST'])
def delete_song(song_name):
    if request.method == 'POST':
        delete_song_from_db(song_name)

        return redirect(url_for('index'))


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


if __name__ == '__main__':
    # server = Server(app.wsgi_app)
    # server.watch('**/*.*')
    # server.serve()
    # load_song_and_duplicate_as_temp('test_song')
    app.run()


