import csv
import pickle
import json
from collections import defaultdict


def read_tweet(path, tokenize_level='word', geo_level=1):
    def word2charngrams(word, n=3):
        return [word[i:i+n] for i in range(len(word)-n+1)]

    user_tweet = dict()
    tweets = []
    id = 0
    with open(path, mode='r') as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_ALL)
        for line in reader:
            tw = line[-1].lower().strip().split()
            tw = [token for token in tw if not token.startswith('@') and not token.startswith('http')]
            if len(tw) <= 10:
                continue
            if not line[1]:
                continue

            if tokenize_level == '3grams':
                tw = word2charngrams(' '.join(tw))

            user = int(line[1])
            lat = float('{:.{prec}f}'.format(float(line[3]), prec=geo_level))
            lon = float('{:.{prec}f}'.format(float(line[4]), prec=geo_level))
            latlon = 'lat|'+str(lat)+'|lon|'+str(lon)
            tweets.append([tw, [user], [lat], [lon], [latlon], [id]])
            if str(user) not in user_tweet:
                user_tweet[str(user)] = [(id, tw)]
            else:
                user_tweet[str(user)].append((id, tw))
            id += 1
    return tweets, user_tweet


def construct_mention_network(user_tweets, path, output_filename):
    network = defaultdict(list)
    with open(path, mode='r') as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_ALL)
        for line in reader:
            if len(line) < 3:
                continue
            user = str(line[1])
            connected_user = line[2:]
            for connection in connected_user:
                connection = str(connection)
                if connection not in network[user] and connection in user_tweets:
                    network[user].append(connection)
                if user not in network[connection] and user in user_tweets:
                    network[connection].append(user)

    return network


def load_pickle(path):
    with open(path, mode='rb') as f:
        return pickle.load(f)


def dump_pickle(obj, path):
    with open(path, mode='wb') as w:
        pickle.dump(obj, w)


def write_embedding(id2word, freqs, nn_embedding, use_cuda, filename):
    all_dics = []
    if use_cuda:
        embeddings = nn_embedding.weight.data.cpu().numpy()
    else:
        embeddings = nn_embedding.weight.data.numpy()
    for latlon_id, vec in enumerate(embeddings):
        latlon = id2word[latlon_id]
        # In the case for <UNK> or <PAD> tokens
        if '|' not in latlon:
            continue
        lat = latlon.split('|')[1]
        lon = latlon.split('|')[3]
        dic = {'lat': float(lat),
               'lon': float(lon),
               'emb': vec.tolist(),
               'population': freqs[latlon]}
        all_dics.append(dic)
    with open(filename, 'w') as fin:
        json.dump(all_dics, fin)
