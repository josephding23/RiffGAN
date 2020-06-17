from flask import Flask
from flask import render_template, url_for, request, redirect
from flask_bootstrap import Bootstrap
from livereload import Server
from web.views.riffs_bp import riffs_bp
from web.views.phrases_bp import phrases_bp
from web.views.tracks_bp import tracks_bp
from web.views.song_bp import song_bp
from web.data.song import tracks

app = Flask(__name__, template_folder='templates', static_folder='static')
app.register_blueprint(riffs_bp, url_prefix='/riffs')
app.register_blueprint(phrases_bp, url_prefix='/phrases')
app.register_blueprint(tracks_bp, url_prefix='/tracks')
app.register_blueprint(song_bp, url_prefix='/song')
Bootstrap(app)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    # server = Server(app.wsgi_app)
    # server.watch('**/*.*')
    # server.serve()
    app.run()


