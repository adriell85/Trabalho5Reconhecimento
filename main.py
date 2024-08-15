import matplotlib
matplotlib.use('TkAgg')
from runs import KNNRuns,DMCRuns,BayesianGaussianDiscriminantRuns,KmeansQuantRuns


def main():
    KmeansQuantRuns(0)
    KmeansQuantRuns(1)
    KmeansQuantRuns(2)

if __name__ == "__main__":
    main()
