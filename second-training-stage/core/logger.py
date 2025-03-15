class Logger:
    def __init__(self):
        self.summary = {
            'Loss/pi': [],
            'Loss/v': [],
            'Loss/entropy': [],
            'Loss/kl': [],
            'Score/explore': [],
            'Score/test': [],
            'Loss': []
        }

    def write(self, summary):
        for key, value in summary.items():
            self.summary[key].append(value)