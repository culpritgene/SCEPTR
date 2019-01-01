from imports import *

def make_table(X,y, steps=42, offset_a=0, offset_b=0, regressor=False):
    table = []
    zeros = np.zeros(y.shape[0])
    for d in np.linspace(X.min()+offset_a, X.max()-offset_b, steps):
        p = zeros.copy()
        if regressor:
            p[np.where(X>d)]=1
        else:
            p[np.where(X<d)]=1
        true_positive = p[np.where(y==1)].sum() #.mean()
        true_negative = len(p[np.where(y==0)])-p[np.where(y==0)].sum() #.mean()
        false_positive = len(y[np.where(p==1)])-y[np.where(p==1)].sum() #.mean()
        false_negative = y[np.where(p==0)].sum() #.mean()
        table.append( [d,true_positive,true_negative,false_positive,false_negative])
    table = pd.DataFrame(table, columns=['dist', 'TP','TN', 'FP', 'FN'])
    return table

def plot_PL(table, ax, title=None, curve=False, **kwargs):
    if curve:
        return (table['TP']/(table['TP']+table['FN']), table['TP']/(table['TP']+table['FP']))
    line, = ax.plot(table['TP']/(table['TP']+table['FN']), table['TP']/(table['TP']+table['FP']), **kwargs)
    line.set_label(title)
    return line,

def plot_ROC(table, ax, title=None, curve=False, **kwargs):
    if curve:
        return (table['FP']/(table['FP']+table['TN']), table['TP']/(table['TP']+table['FN']))
    line, = plt.plot(table['FP']/(table['FP']+table['TN']), table['TP']/(table['TP']+table['FN']),  **kwargs)
    line.set_label(title)
    return line,

def calc_area(rc, pr):
    return np.trapz( x=[0]+list(rc[:-1][::-1])+[1], y=[1]+list(pr[:-1][::-1])+[0]  )

def plot_one_reg(X, y, ax, title, **kwargs):
    tb = make_table(X, y, steps=steps, regressor=True)
    rc, pr = plot_PL(tb, ax, curve=True)
    ar1 = calc_area(rc, pr)
    line, = plot_PL(tb, ax, title=title+'_'+str(round(ar1,2) ), **kwargs)
    return line