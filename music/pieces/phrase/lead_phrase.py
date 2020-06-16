

class LeadPhrase(Phrase):
    def __init__(self, length, tonality, bpm, instr):
        Phrase.__init__(self, length, bpm)

        self.tonic, self.mode = tonality
        self.root_note = note_name_to_num(self.tonic)

        self.instr = instr
