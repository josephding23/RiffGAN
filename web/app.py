from flask import Flask
from flask import render_template, url_for, request, redirect
from flask_bootstrap import Bootstrap
from livereload import Server
from music.pieces.song import *

app = Flask(__name__, template_folder='templates', static_folder='static')
Bootstrap(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/all_riffs')
def all_riffs():
    return render_template('riffs/all_riffs.html', riffs=riffs)


@app.route('/delete_riff/<riff_type>/<riff_no>', methods=['POST'])
def delete_riff(riff_type, riff_no):
    riffs[riff_type].pop(int(riff_no)-1)
    return redirect(url_for('riffs', riff_type='griff', riffs=riffs['griff']))


@app.route('/edit_riff/<riff_type>/<riff_no>', methods=['POST'])
def edit_riff(riff_type, riff_no):
    print(riff_no)
    if request.method == 'POST':
        if riff_type in ['griff', 'briff']:
            raw_length = request.form['edit_length_input']
            raw_degrees_and_types = request.form['edit_degrees_types_input']
            raw_timestamps = request.form['edit_timestamps_input']

            try:
                length = int(raw_length)
            except Exception:
                error = 'Length must be integer'
                return render_template('riffs/griff.html', riffs=riffs[riff_type], riff_type=riff_type, error=error)

            try:
                degrees_and_types = get_degrees_and_types_from_raw(raw_degrees_and_types)
                print(degrees_and_types)
            except Exception:
                error = 'Invalid Degrees & Types format.'
                return render_template('riffs/griff.html', riffs=riffs[riff_type], riff_type=riff_type, error=error)

            try:
                timestamps = get_timestamps_from_raw(raw_timestamps)
                print(timestamps)
            except Exception:
                error = 'Invalid Timestamps format.'
                return render_template('riffs/griff.html', riffs=riffs[riff_type], riff_type=riff_type, error=error)

            riffs[riff_type][int(riff_no)-1] = {
                    'length': length,
                    'degrees_and_types': degrees_and_types,
                    'time_stamps': timestamps,
                    'raw_degrees_and_types': raw_degrees_and_types,
                    'raw_timestamps': raw_timestamps
                }

            return redirect(url_for('riffs', riff_type=riff_type, riffs=riffs[riff_type]))


@app.route('/new_riff/<riff_type>', methods=['POST'])
def new_riff(riff_type):
    if request.method == 'POST':
        if riff_type in ['griff', 'briff']:
            raw_length = request.form['new_length_input']
            raw_degrees_and_types = request.form['new_degrees_types_input']
            raw_timestamps = request.form['new_timestamps_input']

            try:
                length = int(raw_length)
            except Exception:
                error = 'Length must be integer'
                return render_template('riffs/griff.html', riffs=riffs[riff_type], riff_type=riff_type, error=error)

            try:
                degrees_and_types = get_degrees_and_types_from_raw(raw_degrees_and_types)
            except Exception:
                error = 'Invalid Degrees & Types format.'
                return render_template('riffs/griff.html', riffs=riffs[riff_type], riff_type=riff_type, error=error)

            try:
                timestamps = get_timestamps_from_raw(raw_timestamps)
            except Exception:
                error = 'Invalid Timestamps format.'
                return render_template('riffs/griff.html', riffs=riffs[riff_type], riff_type=riff_type, error=error)

            riffs[riff_type].append(
                {
                    'length': length,
                    'degrees_and_types': degrees_and_types,
                    'time_stamps': timestamps,
                    'raw_degrees_and_types': raw_degrees_and_types,
                    'raw_timestamps': raw_timestamps
                 }
            )

            return redirect(url_for('riffs', riff_type=riff_type, riffs=riffs[riff_type]))


@app.route('/riffs/<riff_type>', methods=['GET'])
def riffs(riff_type):
    if riff_type == 'griff':
        return render_template('riffs/griff.html', riffs=riffs['griff'], riff_type='griff')

    elif riff_type == 'briff':
        return render_template('riffs/briff.html', riffs=riffs['briff'], riff_type='briff')

    else:
        assert riff_type == 'driff'
        return render_template('riffs/driff.html', riffs=riffs['driff'], riff_type='driff')


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
