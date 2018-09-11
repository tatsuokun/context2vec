def write_embedding(id2word, nn_embedding, use_cuda, filename):
    with open(filename, 'w') as f:
        f.write('{} {}\n'.format(nn_embedding.num_embeddings, nn_embedding.embedding_dim))
        if use_cuda:
            embeddings = nn_embedding.weight.data.cpu().numpy()
        else:
            embeddings = nn_embedding.weight.data.numpy()

        for word_id, vec in enumerate(embeddings):
            word = id2word[word_id]
            vec = ' '.join(list(map(str, vec)))
            f.write('{} {}\n'.format(word, vec))
