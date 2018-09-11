from torchtext import data


class Dataset:
    def __init__(self,
                 sentences: list,
                 batch_size: int,
                 device: int,
                 pad_token='<PAD>',
                 unk_token='<UNK>'):

        self.sentences = sentences
        self.reversed_sentences = [sentence[::-1] for sentence in sentences]
        self.sentence_id = [[i] for i in range(len(sentences))]
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.device = device

        self.sentence_field = data.Field(use_vocab=True,
                                         unk_token=self.unk_token,
                                         pad_token=self.pad_token,
                                         batch_first=True,
                                         include_lengths=True)
        self.reversed_sentence_field = data.Field(use_vocab=True,
                                                  unk_token=self.unk_token,
                                                  pad_token=self.pad_token,
                                                  batch_first=True)
        self.sentence_id_field = data.Field(use_vocab=False, batch_first=True)

        self.sentence_field.build_vocab(sentences, min_freq=0)
        self.vocab = self.reversed_sentence_field.vocab = self.sentence_field.vocab
        if self.pad_token:
            self.pad_index = self.sentence_field.vocab.stoi[self.pad_token]

        self.dataset = self._create_dataset()
        self._set_batch_iter(batch_size)

    def get_raw_sentence(self, sentences):
        return [[self.vocab.itos[idx] for idx in sentence]
                for sentence in sentences]

    def _create_dataset(self):
        _fields = [('sentence', self.sentence_field),
                   ('reversed_sentence', self.reversed_sentence_field),
                   ('id', self.sentence_id_field)]
        items = [[sentence, rev_sentence, id] for sentence, rev_sentence, id
                 in zip(self.sentences, self.reversed_sentences, self.sentence_id)]
        return data.Dataset(self._get_examples(items, _fields), _fields)

    def _get_examples(self, items: list, fields: list):
        return [data.Example.fromlist(item, fields) for item in items]

    def _set_batch_iter(self, batch_size: int):

        def sort(data: data.Dataset) -> int:
            return len(getattr(data, 'sentence'))

        self.batch_iter = data.BucketIterator(dataset=self.dataset,
                                              batch_size=batch_size,
                                              sort_key=sort,
                                              train=True,
                                              repeat=False,
                                              sort_within_batch=True,
                                              device=self.device)
