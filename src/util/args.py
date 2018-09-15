import argparse


def parse_args():

    gpu_id = -1
    parser = argparse.ArgumentParser(prog='src')
    parser.add_argument('--gpu-id', '-g', default=gpu_id, type=int)
    parser.add_argument('--input-file', '-i', default='dataset/sample.txt', type=str,
                        help='specify input file')
    parser.add_argument('--trimfreq', '-t', default=3, type=int,
                        help='minimum frequency for word in training')
    parser.add_argument('--ns_power', '-p', default=0.75, type=float,
                        help='negative sampling power')
    parser.add_argument('--dropout', '-o', default=0.0, type=float,
                        help='nn dropout')
    parser.add_argument('--wordsfile', '-w', default='models/embedding.vec',
                        help='word embeddings output filename')
    parser.add_argument('--modelfile', '-m', default='models/model.param',
                        help='model output filename')
    parser.add_argument('--hidden_size', '-u', default=300, type=int,
                        help='number of units (dimensions) of one context word')
    parser.add_argument('--batch_size', '-b', default=100, type=int,
                        help='learning minibatch size')
    parser.add_argument('--epoch', '-e', default=10, type=int,
                        help='number of epochs to learn')

    return parser.parse_args()
