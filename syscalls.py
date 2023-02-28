import numpy as np
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
import os

cert_or_unm = 'snd-cert' if False else 'snd-unm'
path = os.path.join('.', 'negative-selection', 'syscalls', cert_or_unm)
jarfile = os.path.join('.', 'negative-selection', 'negsel2.jar')

filename = rf'{path}\{cert_or_unm}'
trainfile = f'{filename}-proc.train'
v = 1
testfile = f'{filename}.{v}-proc.test'
outfile = f'{filename}.{v}.out'

n = 12
r = 4


def preprocess(file):
    chunk = n
    step = 1
    result = ""
    with open(file) as f:
        for line in f.read().split():
            if len(line) < chunk:
                result += line*(chunk//len(line))+line[:chunk % len(line)]+'\n'

            i = 0
            while i+chunk <= len(line):
                result += line[i:i+chunk]+'\n'
                i += step
            result += '\n'

    return result[:-1]


def plot_base():

    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.grid(alpha=0.5)
    plt.title(f'natural selection with n={n}')
    plt.legend(loc='lower right')
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    # preprocess train and test
    with open(trainfile, 'w') as out:
        out.write(preprocess(f'{filename}.train'))

    with open(testfile, 'w') as out:
        out.write(preprocess(f'{filename}.{v}.test'))

    ax = plt.gca()
    for r in range(3, 9):
        # negative selection
        os.system(
            f'cmd /c "java -jar {jarfile} -self {trainfile} -alphabet file://{filename}.alpha -n {n} -r {r} -c -l < {testfile} > {outfile}')

        # post process output
        with open(outfile) as file:
            lines = file.read().split('NaN')
            pred = []
            for l in lines:
                scores = np.array(l.split(), dtype=np.float)
                pred.append(np.max(scores))

        pred = np.array(pred)

        with open(f'{filename}.{v}.labels') as file:
            label = np.array(file.read().split(), dtype=np.float)

        RocCurveDisplay.from_predictions(
            label, pred, name=f'negative selection (r={r})', ax=ax)

    plot_base()
