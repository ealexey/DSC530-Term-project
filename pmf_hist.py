import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

class pmf:
    def __init__(self, val):
        self.val = val

    def key(self, val):
        hist = dict()
        for value in self.val:
            hist[value] = hist.get(value, 0) + 1
        pmf = dict()
        for value, frequency in hist.items():
            pmf[value] = round(frequency / len(val), 2)
        return list(pmf.keys())

    def freq(self, val):
        hist = dict()
        for value in self.val:
            hist[value] = hist.get(value, 0) + 1
        return list(hist.values())

    def prob(self, val):
        hist = dict()
        for value in self.val:
            hist[value] = hist.get(value, 0) + 1
        pmf = dict()
        for value, frequency in hist.items():
            pmf[value] = round(frequency / len(self.val), 2)
        return list(pmf.values())

    def hist(self, val):
        hist = dict()
        for value in self.val:
            hist[value] = hist.get(value, 0) + 1
        pmf = dict()
        for value, frequency in hist.items():
            pmf[value] = round(frequency / len(val), 4)
        return pmf

class normpdf():
    def __init__(self, mu=0, sigma=1, label=""):
        self.mu=mu
        self.sigma=sigma
        self.label=label

    def density(self, x):
        return scipy.stats.norm.pdf(x, self.mu, self.sigma)
    def getlinspace(self):
        low, high=self.mu-3*self.sigma, self.mu+3*self.sigma
        return np.linspace(low, high, 100)




# the function hist plots a pmf histogram
def hist(x, xlabel=" "):
    plt.bar(pmf(x).key(x), pmf(x).prob(x))
    plt.ylabel("Probability")
    plt.title("Histogram")
    plt.xlabel(xlabel)
    plt.show()

# the function hist plots a probability histogram with a normal curve
def hist_bar_norm(x, xlabel=" "):
    mu=x.mean()
    sigma=x.std()
    y = np.linspace(mu - 3*sigma, mu + 3*sigma, 500)
    plt.plot(y, scipy.stats.norm.pdf(y, mu, sigma))
    plt.bar(pmf(x).key(x), pmf(x).prob(x))
    plt.ylabel("Probability")
    plt.title("Histogram")
    plt.xlabel(xlabel)
    plt.show()


# the function hist1 plots a frequency histogram
def hist1(x, xlabel=" ", htitle=" "):
    plt.bar(pmf(x).key(x), pmf(x).freq(x))
    plt.ylabel("Frequency")
    plt.title(htitle)
    plt.xlabel(xlabel)
    plt.show()



# the function hist2 plots two pmf histograms in the same plot
def hist2(x, y, xlabel=" ", title=" ", label=[], axis=[], width=0.25):
    xwidth = [i - width for i in pmf(x).key(x)]
    plt.bar(xwidth, pmf(x).prob(x), align="center", width=width)
    plt.bar(pmf(y).key(y), pmf(y).prob(y), align="center", width=width)
    plt.ylabel("Probability", fontname="Arial", fontsize=16)
    plt.title(title, fontname="Arial", fontsize=20)
    plt.xlabel(xlabel, fontname="Arial", fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(label)
    plt.axis(axis)
    plt.show()


# the function hist3 plots two frequency histograms in the same plot
def hist3(x, y, xlabel=" ", label=[]):
    xwidth = [i - 0.25 for i in pmf(x).key(x)]
    plt.bar(xwidth, pmf(x).freq(x), align="center", width=0.25)
    plt.bar(pmf(y).key(y), pmf(y).freq(y), align="center", width=0.25)
    plt.ylabel("Frequency")
    plt.title("Histogram")
    plt.xlabel(xlabel)
    plt.legend(label)
    plt.show()

def cdfval(x, val):
    count=0
    for i in x:
        if i<=val:
            count+=1
    prob=count/len(x)
    return prob

def cdfx(x):
    x.sort()
    count = 0
    prob = []
    values=[]
    for i in range(len(x)):
        count=count+1
        if i<len(x)-1:
            if x[i]!=x[i+1]:
                values.append(x[i])
                prob.append(count/len(x))
        elif i==len(x)-1:
                values.append(x[len(x)-1])
                prob.append(count/(len(x)))
    print(values)
    return prob

#the cdf_key() function returns the unambiguous, not duplicated values of a initial list (cleaned list)
def cdf_key(x):
    x.sort()
    count = 0
    prob = []
    values=[]
    for i in range(len(x)):
        count=count+1
        if i<len(x)-1:
            if x[i]!=x[i+1]:
                values.append(x[i])
                prob.append(count/len(x))
        elif i==len(x)-1:
                values.append(x[len(x)-1])
                prob.append(count/(len(x)))
    return values

#the function cdf_prob() returns cdf values corresponding to cleaned initial list values
def cdf_prob(x):
    x.sort()
    count = 0
    prob = []
    values=[]
    for i in range(len(x)):
        count=count+1
        if i<len(x)-1:
            if x[i]!=x[i+1]:
                values.append(x[i])
                prob.append(count/len(x))
        elif i==len(x)-1:
                values.append(x[len(x)-1])
                prob.append(count/(len(x)))
    return prob


#step line
def cdf_hist(y, xlabel=" ", htitle=" "):
    x=list()
    for i in y:
        x.append(i)
    x.sort()
    count = 0
    prob = []
    values = []
    for i in range(len(x)):
        count = count + 1
        if i < len(x) - 1:
            if x[i] != x[i + 1]:
                values.append(x[i])
                prob.append(count / len(x))
        elif i == len(x) - 1:
            values.append(x[len(x) - 1])
            prob.append(count / (len(x)))
    values= values + values
    values.sort()
    prob = prob + prob
    prob.sort()
    z = [0]
    prob = z + prob
    del prob[-1]
    plt.plot(values, prob)
    plt.ylabel("CDF")
    plt.title(htitle)
    plt.xlabel(xlabel)
    plt.show()

#complimentary stepline cdf
def ccdf_hist(y, xlabel=" ", htitle=" "):
    x=list()
    for i in y:
        x.append(i)
    x.sort()
    count = 0
    prob = []
    values = []
    for i in range(len(x)):
        count = count + 1
        if i < len(x) - 1:
            if x[i] != x[i + 1]:
                values.append(x[i])
                prob.append(count / len(x))
        elif i == len(x) - 1:
            values.append(x[len(x) - 1])
            prob.append(count / (len(x)))
    values= values + values
    values.sort()
    prob = prob + prob
    prob.sort()
    z = [0]
    prob = z + prob
    del prob[-1]
    prob=[1-i for i in prob]
    plt.plot(values, prob)
    plt.ylabel("CCDF", fontname="Arial", fontsize=16)
    plt.title(htitle, fontname="Arial", fontsize=18)
    plt.xlabel(xlabel, fontname="Arial", fontsize=16)
    plt.yscale("log")
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.show()


#two steplines for cumulative distribution function(cdf)
def cdf__hist2(y, a, xlabel=" ", htitle=" ", label=[]):
    x=list()
    for i in y:
        x.append(i)
    x.sort()
    count = 0
    prob = []
    values = []
    for i in range(len(x)):
        count = count + 1
        if i < len(x) - 1:
            if x[i] != x[i + 1]:
                values.append(x[i])
                prob.append(count / len(x))
        elif i == len(x) - 1:
            values.append(x[len(x) - 1])
            prob.append(count / (len(x)))
    values= values + values
    values.sort()
    prob = prob + prob
    prob.sort()
    z = [0]
    prob = z + prob
    del prob[-1]

    x=list()
    for i in a:
        x.append(i)
    x.sort()
    count = 0
    prob1 = []
    values1 = []
    for i in range(len(x)):
        count = count + 1
        if i < len(x) - 1:
            if x[i] != x[i + 1]:
                values1.append(x[i])
                prob1.append(count / len(x))
        elif i == len(x) - 1:
            values1.append(x[len(x) - 1])
            prob1.append(count / (len(x)))
    values1= values1 + values1
    values1.sort()
    prob1 = prob1 + prob1
    prob1.sort()
    z = [0]
    prob1 = z + prob1
    del prob1[-1]
    plt.plot(values, prob)
    plt.plot(values1, prob1)
    plt.title(htitle, fontname="Arial", fontsize=18)
    plt.ylabel("CDF", fontname="Arial", fontsize=16)
    plt.xlabel(xlabel, fontname="Arial", fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(label)
    plt.show()

def Corr(x, y):
    x=np.asarray(x)
    y=np.asarray(y)
    xmean=x.mean()
    xstd=x.std()
    ymean=y.mean()
    ystd=y.std()
    x=x-xmean
    y=y-ymean
    # x=[i-xmean for i in x]
    # y=[i-ymean for i in y]
    x=np.asarray(x)
    y=np.asarray(y)
    z=x*y
    corr=sum(z)/(len(x)*xstd*ystd)
    return corr


def ave_class_size(d):
    total_students=0
    classes=0
    for size, count in d.items():
        total_students=total_students+size*count
        classes=classes+count
    ave=total_students/classes
    return ave
# print(ave_class_size(d))

def biased_class_size(d):
    total_students=0
    classes_biased=0
    for size, count in d.items():
        classes_biased=classes_biased+size*count*size
        total_students=total_students+size*count
    ave_biased=classes_biased/total_students
    print(classes_biased)
    return ave_biased
# ave_biased=(biased_class_size(d))
# print(ave_biased)

def ave_size(d):
    total=0
    values=0
    for size, count in d.items():
        total=total+size*count
        values=values+count
    ave=total/values
    return ave
# print(ave_size(d))

def ave_biased_size(d):
    total=0
    biased=0
    for size, count in d.items():
        biased=biased+size*count*size
        total=total+size*count
    ave_biased=biased/total
    return ave_biased
# ave_biased=(ave_biased_size(d))
# print("This is new function", ave_biased)

def biased_class_size(d):
    total_students=0
    classes_biased=0
    for size, count in d.items():
        classes_biased=classes_biased+size*count*size
        total_students=total_students+size*count
    ave_biased=classes_biased/total_students
    print(classes_biased)
    return ave_biased
# ave_biased=(biased_class_size(d))
# print(ave_biased)

def unbiased_class_size(ave_biased, d):
    classes_biased = 0
    classes = 0
    for size, count in d.items():
        classes_biased=classes_biased+size*count*size
        classes = classes + count
    ave=classes_biased/(classes*ave_biased)
    return ave
# print(unbiased_class_size(ave_biased, d))

def Percentile2(x, p):
    y=cdfx(x)
    prob=y[0]
    values=y[1]
    for i in range(len(prob)):
        print(i)
        if prob[i]>=p:
            break
    return values[i]

def perc_rank(mylist, x):
    count=0
    for i in mylist:
        if i<x:
            count+=1
    percentile_rank=100*count/len(mylist)
    return percentile_rank

def cov(x, y):
    x=np.asarray(x)
    y=np.asarray(y)
    xmean=np.mean(x)
    ymean=np.mean(y)
    cov=np.dot(x-xmean, y-ymean)/len(x)
    print("covariance is:", cov)

def p_value_permutation_mean(group1, group2, iter=1000):
    n=len(group1)
    group1=np.array(group1)
    group2=np.array(group2)
    stat_diff=abs(group1.mean()-group2.mean())
    print("The difference in means:", stat_diff)
    t=[]
    pool=np.hstack((group1, group2))
    for i in range (iter):
        np.random.shuffle(pool)
        data=pool[:n], pool[n:]
        diff=data[0].mean()-data[1].mean()
        diff=abs(diff)
        t.append(round(diff, 3))
    filtered=list(filter(lambda x: x>stat_diff, t))
    p_value=len(filtered)/(iter)
    return p_value

def conf_interval_perm_mean(group1, group2, iter=1000, p=0.05):
    n=len(group1)
    group1=np.array(group1)
    group2=np.array(group2)
    t=[]
    pool=np.hstack((group1, group2))
    for i in range (iter):
        np.random.shuffle(pool)
        data=pool[:n], pool[n:]
        diff=data[0].mean()-data[1].mean()
        t.append(round(diff, 3))
    return Percentile2(t, p), Percentile2(t, 1 - p)

def conf_interval_spearman(group1, group2, p=0.05, iter=1000):
    group1 = np.array(group1)
    group2 = np.array(group2)
    t = []
    for i in range(iter):
        np.random.shuffle(group1)
        np.random.shuffle(group2)
        corperm = scipy.stats.spearmanr(group1, group2)[0]
        t.append(round(corperm, 3))
    return Percentile2(t, p), Percentile2(t, 1-p)

def p_value_permutation_spearman(group1, group2, iter=1000):
    group1=np.array(group1)
    group2=np.array(group2)
    cor=(scipy.stats.spearmanr(group1, group2)[0])
    print("Spearman correlation coefficient:", cor)
    cor=abs(cor)
    t=[]
    for i in range (iter):
        np.random.shuffle(group1)
        np.random.shuffle(group2)
        corperm = scipy.stats.spearmanr(group1, group2)[0]
        corperm=abs(corperm)
        t.append(round(corperm, 3))
    filtered=list(filter(lambda x: x>cor, t))
    p_value=len(filtered)/(iter)
    return "p value of the Spearman correlation coefficient is:", p_value

def int_slope(x, y):
    x=np.asarray(x)
    y=np.asarray(y)
    slope=(y.std()/x.std())*Corr(x, y)
    inter=y.mean()-slope*x.mean()
#     print(y.mean())
    return slope, inter

def FitLine(x, inter, slope):
    fit_x = np.sort(x)
    fit_y = inter + slope * fit_x
    return fit_x, fit_y
print(__name__)

if __name__=="__main__":
    print("I am Alexey")