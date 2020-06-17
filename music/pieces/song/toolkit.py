
def refresh_track_info(song_info, track_list, included_tracks):

    tracks = []

    excluded_tracks = []

    for index, track in enumerate(track_list):
        tracks.append(track)
        if index not in included_tracks:
            excluded_tracks.append(index)

    song_info['tracks'] = tracks
    song_info['excluded_track_index'] = excluded_tracks


def get_all_genres():
    genres_list = [
        'Acid rock',
        'Alternative metal',
        'Alternative rock',
        'Art punk',
        'Art rock',
        'Baroque pop',
        'Beach music',
        'Beat music',
        'Big beat',
        'Black metal',
        'Blues rock',
        'Britpop',
        'Comedy rock',
        'Country rock',
        'Death metal',
        'Djent',
        'Doom metal',
        'Dream pop',
        'Electronic rock',
        'Emo',
        'Experimental rock',
        'Folk rock',
        'Funk metal',
        'Funk rock',
        'Garage rock',
        'Glam rock',
        'Gothic rock',
        'Groove metal',
        'Grunge',
        'Hard rock',
        'Hardcore punk',
        'Heavy metal',
        'Indie folk',
        'Indie pop',
        'Indie rock',
        'Industrial metal',
        'Jazz rock',
        'Krautrock',
        'Latin alternative',
        'Math rock',
        'Medieval metal'
        'Neoclassical metal',
        'Neo-progressive rock',
        'New wave'
        'Noise rock',
        'Nu metal',
        'Pop punk'
        'Pop rock',
        'Post-britpop',
        'Post-grunge',
        'Post-metal',
        'Post-punk',
        'Power pop',
        'Power metal',
        'Progressive metal',
        'Progressive rock',
        'Protopunk',
        'Psychedelic funk'
        'Psychedelic rock'
        'Punk rock',
        'Rap metal',
        'Rap rock',
        'Reggae rock',
        'Rock and roll',
        'Rockabilly',
        'Shoegazing',
        'Ska punk',
        'Sludge metal',
        'Soft rock',
        'Southern rock',
        'Surf music',
        'Thrash metal',
        'Viking metal'
    ]
    return genres_list