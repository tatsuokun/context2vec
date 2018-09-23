import toml


class Config:

    def __init__(self, filename: str):

        self.filename = filename
        config = toml.load(self.filename)

        nets = config.get('nets', {})
        self.word_embed_size = int(nets.get('word_embed_size', 300))
        self.hidden_size = int(nets.get('hidden_size', 300))
        self.n_layers = int(nets.get('n_layers', 1))
        self.use_mlp = bool(nets.get('use_mlp', True))
        self.dropout = float(nets.get('dropout', 0.0))

        train = config.get('train', {})
        self.n_epochs = int(train.get('n_epochs', 10))
        self.batch_size = int(train.get('batch_size', 100))
        self.min_freq = int(train.get('min_freq', 1))
        self.ns_power = float(train.get('ns_power', 0.75))
        self.learning_rate = float(train.get('learning_rate', 1e-4))

        mscc = config.get('mscc', {})
        self.question_file = mscc.get('question_file')
        self.answer_file = mscc.get('answer_file')


if __name__ == '__main__':
    config = Config('./config.toml')
    print(config)
