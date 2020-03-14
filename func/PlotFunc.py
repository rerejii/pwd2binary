import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def accuracy_plot(path_cls, title, label, item, xlim=None, ylim=None):
    # df = pd.read_csv(path_cls.make_csv_path('accuracy.csv'))
    df = pd.read_csv(path_cls.make_csv_path('step_accuracy.csv'))
    max_val = round(df[item].max(), 5)
    max_id_x = df[item].idxmax()
    max_step = df.at[max_id_x, 'step']
    df.plot(x='step', y=item, label=label + ' ' + str(max_val) + ' (' + str(max_step) + ')', title=title)
    plt.xlabel("step")
    plt.ylabel("Accuracy")
    tmp = ylim if ylim is not None else [0.955, 0.985]
    plt.ylim(tmp)
    tmp = xlim if xlim is not None else [df['step'].min(), df['step'].max()]
    plt.xlim(tmp)
    plt.legend(title="Data Set", loc='lower right')
    plt.savefig(title+'.png')
