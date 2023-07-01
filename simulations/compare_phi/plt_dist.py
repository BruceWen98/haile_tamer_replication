import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import AL_HT_bounds as AHb
import seaborn as sns
import scipy.stats

OUTPATH = "/Users/brucewen/Desktop/honors_thesis/estimation/simulations/compare_phi/figs/"


def plot_beta_pdf_cdf(param1,param2):
    # observations = np.random.beta(param1,param2,10000)

    sns.set_style('darkgrid')
    plt.figure(figsize=(10,6), tight_layout=True)
    
    

    
    
    X = np.linspace(0, 1, 1000)
    pdf = np.power(X,param1-1) * np.power(1-X,param2-1) / scipy.special.beta(param1,param2)
    plt.subplot(1,2,1)
    plt.scatter(X, pdf, label="PDF".format(param1,param2), color="orange")
    plt.xlabel("Value")
    plt.ylabel("PDF")
    plt.title(r"PDF of Beta ($\alpha={}, \beta={}$)".format(param1,param2))
    plt.legend()
    
    cdf = scipy.special.betainc(param1,param2,X)
    plt.subplot(1,2,2)
    plt.scatter(X, cdf, label="CDF".format(param1,param2), color="blue")
    plt.xlabel("Value")
    plt.ylabel("CDF")
    plt.title(r"CDF of Beta ($\alpha={}, \beta={}$)".format(param1,param2))
    plt.legend()
    
    plt.savefig(OUTPATH + "PdfCdf_beta_{}_{}.png".format(param1,param2))
    
    return

plot_beta_pdf_cdf(0.5,0.5)
plot_beta_pdf_cdf(0.1,1)


def plot_betaBimodal_pdf_cdf(param1,param2,param3,param4):
    sns.set_style('darkgrid')
    plt.figure(figsize=(10,6), tight_layout=True)

    
    X = np.linspace(0, 1, 1000)
    pdf = ( np.power(X,param1-1) * np.power(1-X,param2-1) / scipy.special.beta(param1,param2) )/2 + ( 
            np.power(X,param3-1) * np.power(1-X,param4-1) / scipy.special.beta(param3,param4) )/2
    plt.subplot(1,2,1)
    plt.scatter(X, pdf, label="PDF", color="orange")
    plt.xlabel("Value")
    plt.ylabel("PDF")
    plt.title(r"PDF of Bimodal Beta ($\alpha_1={},\beta_1={},\alpha_2={},\beta_2={}$)".format(param1,param2,param3,param4))
    plt.legend()
    
    cdf = scipy.special.betainc(param1,param2,X)/2 + scipy.special.betainc(param3,param4,X)/2
    plt.subplot(1,2,2)
    plt.scatter(X, cdf, label="CDF", color="blue")
    plt.xlabel("Value")
    plt.ylabel("CDF")
    plt.title(r"CDF of Bimodal Beta ($\alpha_1={},\beta_1={},\alpha_2={},\beta_2={}$)".format(param1,param2,param3,param4))
    plt.legend()
    
    plt.savefig(OUTPATH + "PdfCdf_betaBimodal_{}_{}_{}_{}.png".format(param1,param2,param3,param4))
    return
    
plot_betaBimodal_pdf_cdf(2,20,20,2)