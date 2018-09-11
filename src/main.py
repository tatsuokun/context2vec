import os
import torch
# from torch import optim
from src.util.args import parse_args
from src.util.batch import Dataset
from src.nets import Context2vec


def main():
    corpus_filename = 'dataset/sample.txt'

    args = parse_args()
    gpu_id = args.gpu_id
    batch_size = args.batch_size
    n_epochs = args.epoch
    word_embed_size = args.hidden_size
    hidden_size = args.hidden_size
    use_cuda = torch.cuda.is_available() and gpu_id > -1
    if use_cuda:
        device = torch.device('cuda:{}'.format(gpu_id))
        torch.cuda.set_device(gpu_id)
    else:
        device = torch.device('cpu')

    if not os.path.isfile(corpus_filename):
        raise FileNotFoundError
    with open(corpus_filename) as f:
        sentences = [line.strip().lower().split() for line in f]

    print(batch_size, n_epochs, word_embed_size, hidden_size, device)
    dataset = Dataset(sentences, batch_size, device)
    print(dataset)
    model = Context2vec(vocab_size=len(dataset.vocab),
                        counter=[_ for _ in range(10)],
                        word_embed_size=word_embed_size,
                        hidden_size=hidden_size,
                        n_layers=1,
                        bidirectional=True,
                        dropout=0.0,
                        pad_index=dataset.pad_index,
                        inference=False)
    if use_cuda:
        model.cuda()
    print(model)
    for batch in dataset.batch_iter:
        sentence, lengths = getattr(batch, 'sentence')
        rev_sentence = getattr(batch, 'reversed_sentence')
        loss = model(sentence, rev_sentence, lengths)
        print(loss)

    '''

    model = Baseline(vocab_size=len(dataset.vocab),
                     n_locations=len(dataset.latlon_str_field.vocab),
                     word_embed_size=word_embed_size,
                     hidden_size=hidden_size,
                     n_layers=3,
                     bidirectional=True,
                     dropout=0.3,
                     pad_index=dataset.pad_index,
                     tie_weights=False,
                     inference=False)
    if use_cuda:
        model = model.cuda()

    optim_params = model.parameters()
    optimizer = optim.Adam(optim_params, lr=10**-3)

    for epoch in range(n_epochs):
        total_loss = 0
        for batch in dataset.batch_iter:
            tokens_a = getattr(batch, 'tweet')
            geolocation_a = getattr(batch, 'latlon')
            lat = getattr(batch, 'lat')
            lon = getattr(batch, 'lon')
            users = getattr(batch, 'user')
            if use_user_connection:
                users_in_batch = users.cpu().numpy() if use_cuda else users.numpy()
                tokens_b = []
                geolocation_b = []
                user_location = []
                label = []
                for user in users_in_batch:
                    user = str(user[0])
                    if mention_network[user]:
                        pos_user = mention_network[user][np.random.randint(0, len(mention_network[user]))]
                        pos_tweet_id = user_tweet[pos_user][np.random.randint(0, len(user_tweet[pos_user]))][0]
                        pos_tweet = dataset.lookup('tweet', pos_tweet_id)
                        pos_latlon_str = dataset.lookup('latlon_str', pos_tweet_id)
                        pos_latlon = dataset.lookup('latlon', pos_tweet_id)
                        tokens_b.append(pos_tweet)
                        geolocation_b.append(pos_latlon_str)
                        user_location.append(pos_latlon)
                        label.append(1)
                    else:
                        # rand = np.random.randint(0, n_users, negatives)
                        # neg_users = all_users[rand]
                        # neg_tweets_ids = [user_tweet[neg_user][np.random.randint(0, len(user_tweet[neg_user]))][0]
                        #                   for neg_user in neg_users]
                        rand = np.random.randint(0, n_users)
                        neg_user = all_users[rand]
                        neg_tweet_id = user_tweet[neg_user][np.random.randint(0, len(user_tweet[neg_user]))][0]
                        neg_tweet = dataset.lookup('tweet', neg_tweet_id)
                        neg_latlon_str = dataset.lookup('latlon_str', neg_tweet_id)
                        neg_latlon = dataset.lookup('latlon', neg_tweet_id)
                        tokens_b.append(neg_tweet)
                        geolocation_b.append(neg_latlon_str)
                        user_location.append(neg_latlon)
                        label.append(0)

            tokens_b = dataset.padding(tokens_b)
            tokens_b = torch.tensor(tokens_b, dtype=torch.long, device=device)
            geolocation_b = torch.tensor(geolocation_b, dtype=torch.long, device=device)
            latlon_a = torch.cat((lat, lon), dim=1)
            latlon_b = torch.tensor(user_location, dtype=torch.float, device=device)
            connection_label = torch.tensor(label, dtype=torch.long, device=device)

            loss = model(tokens_a,
                         geolocation_a,
                         latlon_a,
                         tokens_b,
                         geolocation_b,
                         latlon_b,
                         connection_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.data.mean()
        print(total_loss.item())

    write_embedding(dataset.latlon_str_field.vocab.itos,
                    dataset.latlon_str_field.vocab.freqs,
                    model.geoemb,
                    use_cuda,
                    'test_latlon.vec.json')
    '''
