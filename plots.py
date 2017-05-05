import matplotlib.pyplot as plot

import numpy as np


def plot_values(values, name='performance', title_x='epochs'):
    plot.figure(name)
    for k in values:
        number_of_epochs = len(values[k])

        epochs = np.arange(0, number_of_epochs, 1)

        plot.axis([0, number_of_epochs + 1, 0.9 * min(values[k]), 1.05 * max(values[k])])
        plot.plot(epochs, values[k], label=k)

        plot.xticks(np.arange(0, number_of_epochs + 1, max(int(number_of_epochs / 8), 1)))
        plot.xlabel(title_x)
        plot.grid(True)
        plot.legend(loc='best')
        plot.savefig(name + '.png', dpi=125)


def extract(text, line):
    if text in line:
        ln = line.split(text)[1].strip()
        if 'is' in ln:
            ln = ln.split('is')[1].strip()
        return float(ln)
    return -1


def plot_acc_f1(log_file_name, model_name):
    accuracies = {'Accuracy': []}
    f_measures = {'F1': [], 'Precision': [], 'Recall': []}

    with open(log_file_name) as f:
        for line in f:
            acc = extract('Accuracy:', line)
            if acc != -1:
                accuracies['Accuracy'].append(acc)

            prec = extract('prec on Pedestrian:', line)
            if prec != -1:
                f_measures['Precision'].append(prec)

            rec = extract('recall on Pedestrian:', line)
            if rec != -1:
                f_measures['Recall'].append(rec)

            f1 = extract('f on Pedestrian:', line)
            if f1 != -1:
                f_measures['F1'].append(f1)

    print("Model %s, best accuracy %f, epoch %d, best f1 %f, epoch %d"
          % (model_name, max(accuracies['Accuracy']), accuracies['Accuracy'].index(max(accuracies['Accuracy'])),
             max(f_measures['F1']), f_measures['F1'].index(max(f_measures['F1']))))
    plot_values(accuracies, name='%s accuracy' % model_name)
    plot_values(f_measures, name='%s f1' % model_name)


def plot_scores(file_name, model_name):
    scores = {'Loss': []}
    i = 0
    with open(file_name) as f:
        for line in f:
            loss = extract('Averaged score:', line)
            if loss != -1:
                i += 1
                if i % 3 == 0:
                    scores['Loss'].append(loss)

    plot_values(scores, name='%s loss' % model_name, title_x='Iterations')



