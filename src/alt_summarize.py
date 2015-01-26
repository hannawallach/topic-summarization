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

    counts = defaultdict(lambda: defaultdict(lambda: zeros(num_topics)))

    for _, row in state.iterrows():
        counts[1][tuple([row['word']])][row['topic']] += 1
        if row['topic'] == prev_topic and row['doc'] == prev_doc:
            ngram.append(row['word'])
        else:
            if len(ngram) > 1 and len(ngram) <= max_phrase_len:
                counts[len(ngram)][tuple(ngram)][prev_topic] += 1
            ngram = [row['word']]
            prev_doc = row['doc']
            prev_topic = row['topic']
    if len(ngram) > 1 and len(ngram) <= max_phrase_len:
        counts[len(ngram)][tuple(ngram)][prev_topic] += 1

    for l in counts:

        ngrams = DataFrame.from_records([[' '.join(x)] + y.tolist() +
                                         [y.sum()] for x, y in
                                         counts[l].items()],
                                        columns=(['ngram'] +
                                                 range(num_topics) + ['same']))
        counts[l] = ngrams

        print >> sys.stderr, 'Selecting %d-gram phrases...' % l

        phrases[l] = set(ngrams[ngrams['same'] >= min_phrase_count]['ngram'])

    scores = defaultdict(lambda: defaultdict(float))

    for l in counts:

        ngrams = counts[l]
        n = ngrams['same'].sum()
        ngrams['prob'] = ngrams['same'] / n

        for topic in xrange(num_topics):

            n_topic = ngrams[topic].sum()
            p_topic = topics['prob'][topic]
            p_not_topic = 1.0 - p_topic

            for _, row in ngrams[(ngrams['ngram'].isin(phrases[l])) &
                                   (ngrams[topic] > 0)].iterrows():

                p_phrase = row['prob']
                p_topic_g_phrase = row[topic] / row['same']
                p_topic_g_not_phrase = ((n_topic - row[topic]) /
                                        (n - row['same']))

                p_not_phrase = 1.0 - p_phrase
                p_not_topic_g_phrase = 1.0 - p_topic_g_phrase
                p_not_topic_g_not_phrase = 1.0 - p_topic_g_not_phrase

                a = 0.0

                if p_topic_g_phrase != 0.0:
                    a += (p_topic_g_phrase *
                          (log2(p_topic_g_phrase) - log2(p_topic)))
                if p_not_topic_g_phrase != 0.0:
                    a += (p_not_topic_g_phrase *
                          (log2(p_not_topic_g_phrase) - log2(p_not_topic)))

                b = 0.0

                if p_topic_g_not_phrase != 0.0:
                    b += (p_topic_g_not_phrase *
                          (log2(p_topic_g_not_phrase) - log2(p_topic)))
                if p_not_topic_g_not_phrase != 0.0:
                    b += (p_not_topic_g_not_phrase *
                          (log2(p_not_topic_g_not_phrase) - log2(p_not_topic)))

                scores[topic][row['ngram']] = p_phrase * a + p_not_phrase * b

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
