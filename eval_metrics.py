import argparse
import os
import sys

from metrics import charades_classify as cc
from metrics import kinetics_classify as kc


class Runnercommands():

    def __init__(self):
        usageText = '''eval_metrics <command> [<args>]

        The following commands are available:
        * kinetics_classify
        * charades_classify '''
        parser = argparse.ArgumentParser(
            description='Generate metrics for model evaluation results.', usage=usageText)

        parser.add_argument('command', help='Subcommand to run.')
        args = parser.parse_args(sys.argv[1:2])

        # Calling specific commands
        getattr(self, args.command)()

    def kinetics_classify(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--log_file', type=str)
        args = parser.parse_args(sys.argv[2:])

        output_file = get_results_file_name(args.log_file)

        labels, preditions, is_causal = kc.read_file(args.log_file)
        if is_causal:
            kc.save_causal(preditions, labels, output_file)
        else:
            kc.save(preditions, labels, output_file)

    def charades_classify(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--log_file', type=str)
        parser.add_argument('--gt_file', type=str)
        args = parser.parse_args(sys.argv[2:])

        output_file = get_results_file_name(args.log_file)
        cc.save(args.log_file, args.gt_file, output_file)


def get_results_file_name(file_name):
    base, file_name = os.path.split(file_name)
    name, _ = os.path.splitext(file_name)
    output_file = os.path.join(base, name+'_results.txt')

    return output_file


if __name__ == '__main__':
    Runnercommands()
