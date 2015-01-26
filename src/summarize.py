from __future__ import division

import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import defaultdict
from numpy import abs, exp, log, log2, percentile, power, zeros
from pandas import DataFrame, Series
from pandas.io.parsers import read_csv
from scipy.special import gammaln

from IPython import embed


def bfc(a, b, c, d, n, a_plus_b, a_plus_c, alpha, alpha_sum):
    """
    """

    num = (gammaln(n + alpha_sum) +
           4 * gammaln(alpha) +
           gammaln(a_plus_b + 2 * alpha - 1.0) +
           gammaln(c + d + 2 * alpha - 1.0) +
           gammaln(a_plus_c + 2 * alpha - 1.0) +
           gammaln(b + d + 2 * alpha - 1.0) +
           2 * gammaln(alpha_sum - 2.0))
    den = (gammaln(alpha_sum) +
           sum([gammaln(alpha + x) for x in [a, b, c, d]]) +
           2 * gammaln(n + alpha_sum - 2.0) +
           4 * gammaln(2 * alpha - 1.0))

    return exp(num - den)


def bfu(a, b, c, d, n, a_plus_b, a_plus_c, alpha, alpha_sum, beta):
    """
    """

    num = (log(1.0 + 1.0 / beta) +
           gammaln(n + alpha_sum - 1.0) +
           4 * gammaln(alpha) +
           gammaln(a_plus_b + 2 * alpha - 1.0) +
           gammaln(c + d + 2 * alpha - 1.0) +
           gammaln(a_plus_c + 2 * alpha - 1.0) +
           gammaln(b + d + 2 * alpha - 1.0) +
           2 * gammaln(alpha_sum - 2.0))
    den = (gammaln(alpha_sum - 1.0) +
           sum([gammaln(alpha + x) for x in [a, b, c, d]]) +
           2 * gammaln(n + alpha_sum - 2.0) +
           4 * gammaln(2 * alpha - 1.0))

    return exp(num - den)


def csy(a, b, c, d, n, a_plus_b, a_plus_c):
    """
    """

    num = n * power(abs(a * d - b * c) - n / 2.0, 2)
    den = ((a_plus_b) * (c + d) * (a_plus_c) * (b + d))

    return num / den


def summarize_topics(filenames, test, selection, dist, max_phrase_len,
                     min_phrase_count):
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

    ngram = dict([(l, l * ['']) for l in xrange(1, max_phrase_len + 1)])
    doc = dict([(l, l * [-1]) for l in xrange(1, max_phrase_len + 1)])
    topic = dict([(l, l * [-1]) for l in xrange(1, max_phrase_len + 1)])

    counts = dict([(l, defaultdict(lambda: zeros(num_topics + 2, dtype=int)))
                   for l in xrange(1, max_phrase_len + 1)])

    for _, row in state.iterrows():
        for l in xrange(1, max_phrase_len + 1):

            ngram[l] = ngram[l][1:] + [row['word']]
            doc[l] = doc[l][1:] + [row['doc']]
            topic[l] = topic[l][1:] + [row['topic']]

            if len(set(doc[l])) == 1:
                if len(set(topic[l])) == 1:
                    counts[l][tuple(ngram[l])][row['topic']] += 1
                    counts[l][tuple(ngram[l])][num_topics] += 1
                counts[l][tuple(ngram[l])][num_topics + 1] += 1

    for l in xrange(1, max_phrase_len + 1):

        ngrams = DataFrame.from_records([[' '.join(x), ' '.join(x[:-1]),
                                          ' '.join(x[1:])] + y.tolist()
                                         for x, y in counts[l].items()],
                                        columns=(['ngram', 'prefix',
                                                  'suffix'] +
                                                 range(num_topics) +
                                                 ['same', 'all']))
        counts[l] = ngrams

#        tmp = state.groupby('doc')['doc'].count()
#        tmp = (len(state) - tmp[tmp < l].sum() - len(tmp[tmp >= l]) * (l - 1))
#        assert ngrams['all'].sum() == tmp
#        assert (sum(ngrams[range(0, num_topics)].sum(axis=1) ==
#                    ngrams['same']) == len(ngrams))

        print >> sys.stderr, 'Selecting %d-gram phrases...' % l

        if l == 1:
            phrases[l] = set(ngrams[ngrams['all'] >=
                                    min_phrase_count]['ngram'])
            continue

        n = ngrams['all'].sum()

        if test == bfu or test == bfc:
            alpha = 1.0
            alpha_sum = 4 * alpha
            beta = alpha_sum / n

        prefix_cache = ngrams.groupby('prefix')['all'].sum()
        suffix_cache = ngrams.groupby('suffix')['all'].sum()

#        assert prefix_cache.sum() == ngrams['all'].sum()
#        assert suffix_cache.sum() == ngrams['all'].sum()

        scores = len(ngrams) * [None]

        for idx, row in ngrams[ngrams['prefix'].isin(phrases[l-1]) &
                               ngrams['suffix'].isin(phrases[l-1]) &
                               (ngrams['all'] >= min_phrase_count)].iterrows():

            a = row['all']

            a_plus_b = suffix_cache[row['suffix']]
            a_plus_c = prefix_cache[row['prefix']]

            b = a_plus_b - a
            c = a_plus_c - a
            d = n - a_plus_b - c

            args = [a, b, c, d, n, a_plus_b, a_plus_c]

            if test == bfu:
                args += [alpha, alpha_sum, beta]
            elif test == bfc:
                args += [alpha, alpha_sum]

            scores[idx] = test(*args)

        ngrams['score'] = scores

        if test == bfu or test == bfc:
            keep = ngrams['score'] <= (1.0 / 10)
        else:
            keep = ngrams['score'] > 10.83

        if selection == 'none':
            phrases[l] = set(ngrams[keep]['ngram'])
        else:
            if l == 2:
                phrases[l] = dict(ngrams[keep].set_index('ngram')['score'])
            else:
                m = 2 if selection == 'bigram' else l-1
                if test == bfu or test == bfc:
                    tmp = set([k for k, v in phrases[m].items()
                               if v <= percentile(sorted(phrases[m].values(),
                                                         reverse=True),
                                                  (1.0 - 1.0 / 2**l) * 100)])
                else:
                    tmp = set([k for k, v in phrases[m].items()
                               if v >= percentile(sorted(phrases[m].values()),
                                                  (1.0 - 1.0 / 2**l) * 100)])
                if selection == 'bigram':
                    keep &= Series([all([' '.join(bigram) in tmp for bigram in
                                         zip(words, words[1:])]) for words in
                                    [ngram.split() for ngram in
                                     ngrams['ngram']]])
                    phrases[l] = set(ngrams[keep]['ngram'])
                else:
                    keep &= (ngrams['prefix'].isin(tmp) &
                             ngrams['suffix'].isin(tmp))
                    phrases[l] = dict(ngrams[keep].set_index('ngram')['score'])

        ngrams.drop(['prefix', 'suffix', 'score'], axis=1, inplace=True)

    if selection == 'bigram':
        phrases[2] = set(phrases[2].keys())
    elif selection == 'n-1-gram':
        for l in xrange(2, max_phrase_len + 1):
            phrases[l] = set(phrases[l].keys())

    scores = defaultdict(lambda: defaultdict(float))

    for l in xrange(1, max_phrase_len + 1):

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

    tests = {
        'bayes-conditional': bfc,
        'bayes-unconditional': bfu,
        'chi-squared-yates': csy
    }

    p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    p.add_argument('--state', type=str, metavar='<state>', required=True,
                   help='gzipped MALLET state file')
    p.add_argument('--topic-keys', type=str, metavar='<topic-keys>',
                   required=True, help='MALLET topics keys file')
    p.add_argument('--test', metavar='<test>', required=True,
                   choices=['bayes-conditional', 'bayes-unconditional',
                            'chi-squared-yates'],
                   help='hypothesis test for phrase generation')
    p.add_argument('--selection', metavar='<selection>', required=True,
                   choices=['none', 'bigram', 'n-1-gram'],
                   help='additional selection criterion')
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
        summarize_topics([args.state, args.topic_keys], tests[args.test],
                         args.selection, args.dist, args.max_phrase_len,
                         args.min_phrase_count)
    except AssertionError:
        p.print_help()


if __name__ == '__main__':
    main()
