# Deployment Module

import sys
import os
import typing
import warnings

from data import data_ingest
from model import model_q1, model_q2, model_q3, model_q4, model_q5

os.system('cls')
warnings.simplefilter(action='ignore')


def run():  # run the analysis pipeline from data manipulation to modelling and log the process
    class Tee(object):
        def __init__(self, *files):
            self.files = files

        def write(self, obj):
            for file in self.files:
                file.write(obj)
                file.flush()

        def flush(self):
            for file in self.files:
                file.flush()

    # log the outputs for further investigations

    f: typing.TextIO = open('./results/logs.txt', 'w+')

    console = sys.stdout
    sys.stdout = Tee(sys.stdout, f)

    print('\n Technical Assessment \n Data Scientist, Digital Surveillance Collection')
    print('\n Project: Kaggle v1.5')

    # data manipulation

    print('\n', 60 * '-', '\n Data \n', 60 * '-')

    data = data_ingest()

    # modelling and analysis

    print('\n', 60 * '-', '\n Model \n', 60 * '-')

    model_q1(data, verbose=True)
    model_q2(data, verbose=True)
    model_q3(data, verbose=True)
    model_q4(data, verbose=True)
    model_q5(data, verbose=True)

    print('\n The process is finalized successfully. \n')

    sys.stdout = console
    f.close()

    return


if __name__ == '__main__':
    run()
