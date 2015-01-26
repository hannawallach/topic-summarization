from __future__ import division

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import Counter, defaultdict
from pandas.io.parsers import read_csv

from IPython import embed


def convert(state_filename, output_filenames):
    """
    """

    state = read_csv(state_filename, compression='gzip', skiprows=2,
                     usecols=[0, 4, 5], header=0,
                     names=['doc', 'word', 'topic'], sep=' ')
    state['word'] = state['word'].astype(str)

    with file(output_filenames[0], 'wb') as f:
        for doc, df in state.groupby('doc'):
            f.write(' '.join(df['word']) + '\n')

    vocab = state['word'].unique()
    with file(output_filenames[1], 'wb') as f:
        f.write('\n'.join(vocab) + '\n')

    vocab = dict(zip(vocab, xrange(len(vocab))))
    with file(output_filenames[2], 'wb') as f:
        for doc, df in state.groupby('doc'):
            counts = defaultdict(lambda: Counter())
            for _, row in df.iterrows():
                counts[vocab[row['word']]][row['topic']] += 1
            f.write(str(len(counts)))
            for k, v in counts.items():
                f.write(' ' + str(k) + ':' + str(v.most_common(1)[0][0]))
            f.write('\n')


def main():

    p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    p.add_argument('--state', type=str, metavar='<state>', required=True,
                   help='gzipped MALLET state file')
    p.add_argument('--output', type=str, metavar='<output-files>', nargs='+',
                   required=True, help='output files')

    args = p.parse_args()

    try:
        convert(args.state, args.output)
    except AssertionError:
        p.print_help()


if __name__ == '__main__':
    main()
