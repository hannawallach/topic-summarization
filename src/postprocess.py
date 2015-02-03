import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import defaultdict
from glob import glob
from pandas import read_csv


def get_prefix(results):
    """
    """

    if os.path.split(results)[1] == 'topic-keys.txt':
        return os.path.split(os.path.split(results)[0])[1]
    elif 'perm' in results:
        prefix = os.path.split(results)
        if prefix[1] == '':
            prefix = os.path.split(prefix[0])
        return prefix[1]
    else:
        return os.path.splitext(os.path.split(results)[1])[0]


def get_summaries(results):
    """
    """

    summaries = defaultdict(list)

    if os.path.split(results)[1] == 'topic-keys.txt':
        topics = read_csv(results, sep='(?: |\t)', engine='python',
                          index_col=0, header=None,
                          names=(['alpha'] + [x for x in xrange(1, 20)]))
        for topic, row in topics.iterrows():
            summaries[str(topic)] = row[1:11]
    elif 'no-perm' in results:
        for filename in glob(results + '/topic*.txt'):
            topic = int(os.path.splitext(os.path.split(filename)[1])[0][-3:])
            for line in open(filename, 'r'):
                fields = line.split()
                summaries[str(topic)].append(' '.join(fields[:-1]))
                if len(summaries[str(topic)]) == 10:
                    break
    else:
        for line in open(results, 'r'):
            fields = line.split()
            if len(fields) >= 2:
                if fields[0] == 'Topic':
                    topic = fields[1][:-1]
                else:
                    summaries[topic].append(' '.join(fields[:-1]))

    return summaries


def reformat(results, output):
    """
    """

    prefix = get_prefix(results)
    summaries = get_summaries(results)

    if not os.path.exists(output):
        os.makedirs(output)
    for topic, summary in summaries.items():
        with file(os.path.join(output, prefix + '_' + topic + '.txt'),
                  'wb') as f:
            f.write('; '.join(summary) + '\n')


def compute_stats(results, output):
    """
    """

    prefix = get_prefix(results)
    summaries = get_summaries(results)

    counts = defaultdict(int)
    for topic, summary in summaries.items():
        for phrase in summary:
            counts[len(phrase.split())] += 1

    if not os.path.exists(output):
        os.makedirs(output)
    with file(os.path.join(output, prefix + '.txt'), 'wb') as f:
        f.write('n\tcount\n')
        for length, count in counts.items():
            f.write('%d\t%d\n' % (length, count))


def main():

    tasks = {
        'reformat': reformat,
        'compute_stats': compute_stats
    }

    p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    p.add_argument('task', metavar='<task>',
                   choices=['reformat', 'compute_stats'],
                   help='task to perform')
    p.add_argument('--results', type=str, metavar='<results>', required=True,
                   help='results file/directory to process')
    p.add_argument('--output', type=str, metavar='<output>', required=True,
                   help='output file/directory')

    args = p.parse_args()

    try:
        tasks[args.task](args.results, args.output)
    except AssertionError:
        p.print_help()


if __name__ == '__main__':
    main()
