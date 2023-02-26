import numpy as np
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
import os

path = r'.\negative-selection'
jarfile = os.path.join(path, 'negsel2.jar')
selffile = os.path.join(path, 'english.train')
engfile = os.path.join(path, 'english.test')
tagfile = os.path.join(path, 'tagalog.test')

def plot_roc(selfout,foreignout,r):
    with open(selfout) as file:
        self = np.array(file.read().split(), dtype=np.float)
    with open(foreignout) as file:
        foreign = np.array(file.read().split(), dtype=np.float)

    pred = np.concatenate((self, foreign), axis=0)
    label = np.concatenate((np.zeros(self.shape), np.ones(foreign.shape)), axis=0)

    RocCurveDisplay.from_predictions(label, pred,name=f'negative selection (r={r})')
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.grid(alpha=0.5)
    plt.legend(loc='lower right')
    plt.tight_layout()

    plt.show()

def task1_1():
    for r in range(1, 10):
        engout = os.path.join(path, 'out', f'english{r}.out')
        tagout = os.path.join(path, 'out', f'tagalog{r}.out')

        if not os.path.exists(engout):
            os.system(
                f'cmd /c "java -jar {jarfile} -self {selffile} -n 10 -r {r} -c -l < {engfile} > {engout}')
        if not os.path.exists(tagout):
            os.system(
                f'cmd /c "java -jar {jarfile} -self {selffile} -n 10 -r {r} -c -l < {tagout} > {tagout}')
            
        plot_roc(engout,tagout,r)    

   



def task1_3():
    languages = ['hiligaynon','middle-english','plautdietsch','xhosa']
    
    for lang in languages:
        otherfile = os.path.join(path,'lang',f'{lang}.txt')
        engout = os.path.join(path, 'out', f'english3.out')
        other = os.path.join(path,'out',f'{lang}3.out')

        if not os.path.exists(engout):
            os.system(
                f'cmd /c "java -jar {jarfile} -self {selffile} -n 10 -r 3 -c -l < {engfile} > {engout}')
        if not os.path.exists(other):
            os.system(
                f'cmd /c "java -jar {jarfile} -self {selffile} -n 10 -r 3 -c -l < {otherfile} > {other}')

        plot_roc(engout,other,3)

if __name__ == '__main__':
    task1_1()
    task1_3()        


