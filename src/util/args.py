import argparse


def parse_args():

    gpu_id = -1
    parser = argparse.ArgumentParser(prog='src')
    parser.add_argument('--gpu-id', '-g', default=gpu_id, type=int)
    parser.add_argument('--train', '-t', action='store_true',
                        help='train or not')
    parser.add_argument('--input-file', '-i', default='dataset/sample.txt', type=str,
                        help='specify input file')
    parser.add_argument('--config-file', '-c', default='./config.toml', type=str,
                        help='specify config toml file')
    parser.add_argument('--wordsfile', '-w', default='models/embedding.vec',
                        help='word embeddings output filename')
    parser.add_argument('--modelfile', '-m', default='models/model.param',
                        help='model output filename')
    parser.add_argument('--task', default='', type=str,
                        help='choose evaluation task from [mscc]')

    return parser.parse_args()
