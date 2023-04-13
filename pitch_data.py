import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random

df = pd.read_csv("pitches.csv",nrows=10000)
data = df.drop(['code', 'type', 'pitch_type',
       'event_num', 'b_score', 'ab_id', 'b_count', 's_count', 'outs',
       'pitch_num', 'on_1b', 'on_2b', 'on_3b','y0','break_y', 'ax', 'ay', 'az', 'sz_bot',
       'sz_top', 'type_confidence','x', 'x0', 'y', 'z0','zone','end_speed'], axis = 1)
data.fillna(data.mean(), inplace=True)

scale = StandardScaler()
scale_data = scale.fit_transform(data)
pca = PCA(n_components=13)
pca.fit(scale_data)
pitches_pca = pca.transform(scale_data)
perc_exp = pca.explained_variance_ratio_
per_var = np.round(perc_exp*100,decimals=1)
loadings = pca.components_

num_pc = pca.n_features_
pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
loadings_df['variable'] = data.columns.values
loadings_df = loadings_df.set_index('variable')
print(loadings_df)

def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'black', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'black', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.title("PCA Biplot")
    plt.grid()

#Call the function. Use only the 2 PCs.
myplot(pitches_pca[:,0:2],np.transpose(loadings[0:2, :]))

var_sum = 0
count = 0
for i in range(len(perc_exp)):
    if var_sum < .7:
        var_sum+=perc_exp[i]
        count+=1
#print("The variance explained is " + str(var_sum)+ " in " + str(count) + " principal components")

pcaComp_df = pd.DataFrame(data = pca.components_,index = ['px', 'pz', 'start_speed', 'spin_rate', 'spin_dir',
       'break_angle', 'break_length', 'vx0', 'vy0', 'vz0','pfx_x', 'pfx_z', 'nasty'],
       columns = ["PC1","PC2","PC3","PC4", "PC5","PC6","PC7","PC8","PC9",
                            "PC10","PC11","PC12", "PC13"])
pcaComp_df = pcaComp_df.drop(columns = ["PC3","PC4", "PC5","PC6","PC7","PC8","PC9",
                            "PC10","PC11","PC12", "PC13"])

pitches_df = pd.DataFrame(data = pitches_pca,columns=["PC1","PC2","PC3","PC4", "PC5","PC6","PC7","PC8","PC9",
                            "PC10","PC11","PC12", "PC13"])
pitches_df = pitches_df.drop(columns =["PC3","PC4", "PC5","PC6","PC7","PC8","PC9",
                            "PC10","PC11","PC12", "PC13"] )

plt.figure()
plt.plot(np.cumsum(per_var))
plt.ylabel('Percentage of Expalined Variance')
plt.xlabel('Principal Component Index')
plt.title('Variance Explained')

plt.figure(2)
pcaComp_df.plot.scatter(x = "PC1", y = "PC2")
plt.title('PCA Comp')

plt.show()