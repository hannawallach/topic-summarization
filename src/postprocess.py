import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import defaultdict
from glob import glob
from pandas import read_csv


def postprocess(results_filename, output_dirname):
    """
    """

    summaries = defaultdict(list)

    if os.path.split(results_filename)[1] == 'topic-keys.txt':
        prefix = os.path.split(os.path.split(results_filename)[0])[1]
        topics = read_csv(results_filename, sep='(?: |\t)', engine='python',
                          index_col=0, header=None,
                          names=(['alpha'] + [x for x in xrange(1, 20)]))
        for topic, row in topics.iterrows():
            summaries[str(topic)] = row[1:11]
    elif 'no-perm' in results_filename:
        prefix = os.path.split(results_filename)
        if prefix[1] == '':
            prefix = os.path.split(prefix[0])
        prefix = prefix[1]
        for filename in glob(results_filename + '/topic*.txt'):
            topic = int(os.path.splitext(os.path.split(filename)[1])[0][-3:])
            for line in open(filename, 'r'):
                fields = line.split()
                summaries[str(topic)].append(' '.join(fields[:-1]))
                if len(summaries[str(topic)]) == 10:
                    break
    else:
        prefix = os.path.splitext(os.path.split(results_filename)[1])[0]
        for line in open(results_filename, 'r'):
            fields = line.split()
            if len(fields) >= 2:
                if fields[0] == 'Topic':
                    topic = fields[1][:-1]
                else:
                    summaries[topic].append(' '.join(fields[:-1]))

    if not os.path.exists(output_dirname):
        os.makedirs(output_dirname)

    for topic, summary in summaries.items():
        assert len(summary) == 10
        with file(os.path.join(output_dirname, prefix + '_' + topic + '.txt'),
                  'wb') as f:
            f.write('; '.join(summary) + '\n')


def main():

    p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    p.add_argument('--results', type=str, metavar='<results>', required=True,
                   help='results file to process')
    p.add_argument('--output', type=str, metavar='<output>', required=True,
                   help='output directory')

    args = p.parse_args()

    try:
        postprocess(args.results, args.output)
    except AssertionError:
        p.print_help()


if __name__ == '__main__':
    main()
