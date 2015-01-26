from __future__ import division

import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import defaultdict
from numpy import abs, exp, log, log2, percentile, power, zeros
from pandas import DataFrame, Series
from pandas.io.parsers import read_csv
from scipy.special import gammaln

from IPython import embed


def summarize_topics(filenames, dist, max_phrase_len, min_phrase_count):
    """
    """

    state = read_csv(filenames[0], compression='gzip', skiprows=2,
                     usecols=[0, 4, 5], header=0,
                     names=['doc', 'word', 'topic'], sep=' ')
    state['word'] = state['word'].astype(str)

    topics = read_csv(filenames[1], sep='(?: |\t)', engine='python',
                      index_col=0, header=None,
                      names=(['alpha'] + [x for x in xrange(1, 20)]))
    if dist == 'average-posterior':
        topics['prob'] = zeros(len(topics))
        for _, df in state.groupby('doc'):
            topics['prob'] += (topics['alpha'].add(df.groupby('topic').size(),
                                                   fill_value=0) /
                               (topics['alpha'].sum() + len(df)))
        topics['prob'] /= state['doc'].nunique()
    elif dist == 'empirical':
        topics['prob'] = state.groupby('topic')['word'].count() / len(state)
    else:
        topics['prob'] = topics['alpha'] / topics['alpha'].sum()

#    assert topics['prob'].sum() >= 1-1e-15
#    assert topics['prob'].sum() <= 1+1e-15

    num_topics = len(topics)

    phrases = dict()

    print >> sys.stderr, 'Creating candidate n-grams...'

    ngram = []
    prev_doc = -1
    prev_topic = -1

    counts = defaultdict(lambda: defaultdict(int))

    for _, row in state.iterrows():
        if row['topic'] == prev_topic and row['doc'] == prev_doc:
            ngram.append(row['word'])
        else:
            if len(ngram) > 1 and len(ngram) <= max_phrase_len:
                counts[prev_topic][' '.join(ngram)] += 1
            ngram = [row['word']]
            prev_doc = row['doc']
            prev_topic = row['topic']
    if len(ngram) > 1 and len(ngram) <= max_phrase_len:
        counts[prev_topic][' '.join(ngram)] += 1

    scores = defaultdict(lambda: defaultdict(float))

    for topic in xrange(num_topics):
        n_topic = sum(counts[topic].values())
        for ngram, count in counts[topic].items():
                scores[topic][ngram] = count / n_topic

    for topic, row in topics.iterrows():
        print 'Topic %d: %s' % (topic, ' '.join(row[1:11]))
        print '---'
        print '\n'.join(['%s (%f)' % (x, y) for x, y in
                         sorted(scores[topic].items(), key=(lambda x: x[1]),
                                reverse=True)][:10]) + '\n'

    return


def main():

    p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    p.add_argument('--state', type=str, metavar='<state>', required=True,
                   help='gzipped MALLET state file')
    p.add_argument('--topic-keys', type=str, metavar='<topic-keys>',
                   required=True, help='MALLET topics keys file')
    p.add_argument('--dist', metavar='<dist>', required=True,
                   choices=['average-posterior', 'empirical', 'prior'],
                   help='distribution over topics')
    p.add_argument('--max-phrase-len', type=int, metavar='<max-phrase-len>',
                   default=5, help='maximum phrase length')
    p.add_argument('--min-phrase-count', type=int,
                   metavar='<min-phrase-count>',
                   default=15, help='minimum phrase count')

    args = p.parse_args()

    try:
        summarize_topics([args.state, args.topic_keys], args.dist,
                         args.max_phrase_len, args.min_phrase_count)
    except AssertionError:
        p.print_help()


if __name__ == '__main__':
    main()
