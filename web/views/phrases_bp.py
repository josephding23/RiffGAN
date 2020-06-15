from flask import Blueprint, render_template, redirect, url_for, request
from music.pieces.song import *
from music.pieces.toolkit import set_used_riff_num_info

phrases_bp = Blueprint('phrases', __name__, template_folder='templates', static_folder='static', url_prefix='/phrases')

json_path = 'D:/PycharmProjects/RiffGAN/data/pieces/songs/json/test_song.json'

song = create_song_drom_json(json_path)

phrases = song.get_all_phrases()
riffs = song.get_all_riffs()

set_used_riff_num_info(phrases, riffs)


@phrases_bp.route('/<phrase_type>', methods=['GET'])
def get_phrases(phrase_type):
    if phrase_type == 'rhythm_guitar_phrase':
        return render_template('phrases/rhythm_guitar_phrase.html',
                               phrases=phrases['rhythm_guitar_phrase'], phrase_type='rhythm_guitar_phrase')
    elif phrase_type == 'rhythm_bass_phrase':
        return render_template('phrases/rhythm_bass_phrase.html',
                               phrases=phrases['rhythm_bass_phrase'], phrase_type='rhythm_bass_phrase')
    else:
        assert phrase_type == 'drum_phrase'
        return render_template('phrases/drum_phrase.html',
                               phrases=phrases['drum_phrase'], phrase_type='drum_phrase')


@phrases_bp.route('/delete/<phrase_type>/<index>', methods=['POST'])
def delete_phrase(phrase_type, index):
    phrases[phrase_type].pop(int(index)-1)
    return redirect(url_for('phrases.get_phrases', phrase_type=phrase_type))


@phrases_bp.route('/edit/<phrase_type>/<index>', methods=['POST'])
def edit_phrase(phrase_type, index):
    if request.method == 'POST':
        if phrase_type in ['rhythm_guitar_phrase', 'rhythm_bass_phrase']:
            raw_length = request.form['edit_length_input']
            raw_bpm = request.form['edit_bpm_input']
            raw_tonality = request.form['edit_tonality_input']
            raw_instr = request.form['edit_instr_input']
            raw_used_riffs = request.form['edit_used_riffs_input']
            raw_arrangements = request.form['edit_arrangements_input']



            print(raw_instr)
    return redirect(url_for('phrases.get_phrases', phrase_type=phrase_type))


@phrases_bp.route('/new/<phrase_type>', methods=['POST'])
def new_phrase(phrase_type):
    return redirect(url_for('phrases.get_phrases', phrase_type=phrase_type))