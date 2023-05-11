
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def createConfusionMatrix(conf_matrix,classes_):
    # constant for classes
    classes = classes_

    # Build confusion matrix
    cf_matrix = conf_matrix
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    return sn.heatmap(df_cm, annot=True).get_figure()