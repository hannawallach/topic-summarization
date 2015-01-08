from __future__ import division

import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import defaultdict
from numpy import abs, exp, log, log2, power, zeros
from pandas import DataFrame
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


def summarize_topics(filenames, test, max_phrase_len, min_phrase_count):
    """
    """

    state = read_csv(filenames[0], compression='gzip', skiprows=2,
                     usecols=[0, 4, 5], header=0,
                     names=['doc_id', 'word', 'topic'], sep=' ')
    state['word'] = state['word'].astype(str)

    topics = read_csv(filenames[1], sep='(?: |\t)', engine='python',
                      index_col=0, header=None,
                      names=(['alpha'] + [x for x in xrange(1, 20)]))
    topics['prob'] = state.groupby('topic')['word'].count() / len(state)
#    topics['prob'] = topics['alpha'] / topics['alpha'].sum()

    num_topics = len(topics)

    phrases = defaultdict(set)
    counts = defaultdict(DataFrame)

    print >> sys.stderr, 'Creating candidate 1-grams...'

    ngrams = defaultdict(lambda: zeros(num_topics + 1, dtype=int))

    for _, row in state.iterrows():
        ngrams[row['word']][row['topic']] += 1
        ngrams[row['word']][num_topics] += 1

    ngrams = DataFrame.from_records([[x] + y.tolist() + [y[num_topics]]
                                     for x, y in ngrams.items()],
                                    columns=(['ngram'] + range(num_topics) +
                                             ['same', 'all']))

#    assert ngrams['all'].sum() == len(state)
#    assert sum(ngrams['same'] == ngrams['all']) == len(ngrams)
#    assert len(ngrams) == len(state['word'].unique())
#    assert ngrams[range(0, num_topics)].sum().sum() == len(state)

    print >> sys.stderr, 'Selecting 1-gram phrases...'

    phrases[1] = set(ngrams[ngrams['all'] >= min_phrase_count]['ngram'])

    ngrams['prob'] = ngrams['same'] / ngrams['same'].sum()
    counts[1] = ngrams

    for l in xrange(2, max_phrase_len + 1):

        print >> sys.stderr, 'Creating candidate %d-grams...' % l

        ngrams = defaultdict(lambda: zeros(num_topics + 2, dtype=int))

        curr_ngram = l * ['']
        doc_history = l * [-1]
        topic_history = l * [-1]

        for _, row in state.iterrows():

            curr_ngram = curr_ngram[1:] + [row['word']]
            doc_history = doc_history[1:] + [row['doc_id']]
            topic_history = topic_history[1:] + [row['topic']]

            if len(set(doc_history)) == 1:
                if len(set(topic_history)) == 1:
                    ngrams[tuple(curr_ngram)][row['topic']] += 1
                    ngrams[tuple(curr_ngram)][num_topics] += 1
                ngrams[tuple(curr_ngram)][num_topics + 1] += 1

        ngrams = DataFrame.from_records([[' '.join(x), ' '.join(x[:-1]),
                                          ' '.join(x[1:])] + y.tolist()
                                         for x, y in ngrams.items()],
                                        columns=(['ngram', 'prefix',
                                                  'suffix'] +
                                                 range(num_topics) +
                                                 ['same', 'all']))

#        tmp = state.groupby('doc_id')['doc_id'].count()
#        tmp = (len(state) - tmp[tmp < l].sum() - len(tmp[tmp >= l]) * (l - 1))
#        assert ngrams['all'].sum() == tmp
#        assert (sum(ngrams[range(0, num_topics)].sum(axis=1) ==
#                    ngrams['same']) == len(ngrams))

        print >> sys.stderr, 'Selecting %d-gram phrases...' % l

        n = ngrams['all'].sum()

        if test == bfu or test == bfc:
            alpha = 1.0
            alpha_sum = 4 * alpha
            beta = alpha_sum / n

        prefix_cache = ngrams.groupby('prefix')['all'].sum()
        suffix_cache = ngrams.groupby('suffix')['all'].sum()

        assert prefix_cache.sum() == ngrams['all'].sum()
        assert suffix_cache.sum() == ngrams['all'].sum()

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
            phrases[l] = set(ngrams[ngrams['score'] <= (1.0 / 10)]['ngram'])
#            for _, row in ngrams[ngrams['score'] <= (1.0 / 10)].iterrows():
#                print row['ngram'] + '\t' + str(row['score'])
        else:
            phrases[l] = set(ngrams[ngrams['score'] > 10.83]['ngram'])
#            for _, row in ngrams[ngrams['score'] > 10.83].iterrows():
#                print row['ngram'] + '\t' + str(row['score'])

        ngrams.drop(['prefix', 'suffix', 'score'], axis=1, inplace=True)
        ngrams['prob'] = ngrams['same'] / ngrams['same'].sum()
        counts[l] = ngrams

    scores = defaultdict(lambda: defaultdict(float))

    for l in xrange(1, max_phrase_len + 1):

        ngrams = counts[l]

        n = ngrams['same'].sum()

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

    for topic, row in topics.sort('prob', ascending=False).iterrows():
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
                   choices=['bayes-unconditional', 'bayes-conditional',
                            'chi-squared-yates'],
                   help='hypothesis test for phrase generation')
    p.add_argument('--max-phrase-len', type=int, metavar='<max-phrase-len>',
                   default=5, help='maximum phrase length')
    p.add_argument('--min-phrase-count', type=int,
                   metavar='<min-phrase-count>',
                   default=15, help='minimum phrase count')

    args = p.parse_args()

    try:
        summarize_topics([args.state, args.topic_keys], tests[args.test],
                         args.max_phrase_len, args.min_phrase_count)
    except AssertionError:
        p.print_help()


if __name__ == '__main__':
    main()
