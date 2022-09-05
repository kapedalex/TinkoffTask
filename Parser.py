import argparse

from Resources.Constants import Constants


def get_train_args():
    parser = argparse.ArgumentParser(description='Training parser')
    parser.add_argument('--input-dir', type=str, default=Constants.TRAIN_TEXT_FILE_PATH,
                        help='input batch size for training')
    parser.add_argument('--model', type=str, default=Constants.SAVE_PATH_MODEL,
                        help='input batch size for training')
    parser.add_argument('--batch', type=int, default=64,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--length', type=int, default=200,
                        help='prediction length')
    parser.add_argument('--prefix', type=str, default=' ',
                        help='start of a seq')
    parser.add_argument('--seq', type=int, default=256,
                        help='sequence length')
    return parser.parse_args()


def get_generate_args():
    parser = argparse.ArgumentParser(description='Training parser')
    parser.add_argument('--model', type=str, default=Constants.SAVE_PATH_MODEL,
                        help='load path')
    parser.add_argument('--length', type=int, default=200,
                        help='prediction length')
    parser.add_argument('--prefix', type=str, default=' ',
                        help='start of a seq')
    return parser.parse_args()