import os
import sys
import logging
import binascii


def set_logging(to_stdout, log_file=None):
    if to_stdout:
        logging.basicConfig(
            format='%(asctime)s : %(message)s',
            level=logging.INFO,
            stream=sys.stdout)
    else:
        logging.basicConfig(
            format='%(asctime)s : %(message)s',
            level=logging.INFO,
            filename=log_file)
        print("logs could be found at %s" % log_file)
    return


def get_experiment_id():
    experiment_id = binascii.hexlify(os.urandom(10))
    # experiment_name = '_'.join([config['']])
    return experiment_id
