# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random

# create pitches data frame
df = pd.read_csv(".\Data\pitches.csv", nrows=10000)
pdata = df.drop(
    [
        "code",
        "type",
        "px",
        "pz",
        "event_num",
        "b_score",
        "ab_id",
        "b_count",
        "s_count",
        "outs",
        "pitch_num",
        "on_1b",
        "on_2b",
        "on_3b",
        "y0",
        "break_y",
        "ax",
        "ay",
        "az",
        "sz_bot",
        "sz_top",
        "type_confidence",
        "x",
        "x0",
        "y",
        "z0",
        "zone",
        "end_speed",
    ],
    axis=1,
)

# drop missing values
pdata = pdata.dropna()

# reclassify pitch type as fastballs and offspeed
pdata["pitch_type"] = pdata["pitch_type"].replace(["FC", "FF", "FT", "SI"], 0)
pdata["pitch_type"] = pdata["pitch_type"].replace(["CH", "CU", "FS", "KC", "SL"], 1)
pdata["pitch_type"] = pdata["pitch_type"].replace(
    ["EP", "FO", "IN", "KN", "PO", "UN"], 2
)

# drop NA pitch type and targets from data
xdata = pdata[pdata["pitch_type"] < 2]
targets = xdata["pitch_type"].to_numpy()
data = xdata.drop({"pitch_type"}, axis=1)

# intialize PCA function
scale = StandardScaler()
scale_data = scale.fit_transform(data)
pca = PCA(n_components=11)
pca.fit(scale_data)
pitches_pca = pca.transform(scale_data)
perc_exp = pca.explained_variance_ratio_
per_var = np.round(perc_exp * 100, decimals=1)
loadings = pca.components_

# create features of each principal component
num_pc = pca.n_features_
pc_list = ["PC" + str(i) for i in list(range(1, num_pc + 1))]
loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
loadings_df["variable"] = data.columns.values
loadings_df = loadings_df.set_index("variable")


# create PCA biplot
def myplot(score, coeff, y, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    # creale color legend from pitch type
    classes = np.unique(y)
    colors = ["g", "r"]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    xset = xs * scalex
    yset = ys * scaley
    # create scatter plot
    for s, l in enumerate(classes):
        plt.scatter(xset[y == l], yset[y == l], c=colors[s])
    # create biplot variable arrows
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color="black", alpha=0.5)
        if labels is None:
            plt.text(
                coeff[i, 0] * 1.15,
                coeff[i, 1] * 1.15,
                "Var" + str(i + 1),
                color="black",
                ha="center",
                va="center",
            )
        else:
            plt.text(
                coeff[i, 0] * 1.15,
                coeff[i, 1] * 1.15,
                labels[i],
                color="black",
                ha="center",
                va="center",
            )
    # format biplot
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.title("PCA Biplot")
    plt.grid()


# Call the function using only the first 2 PCs.
myplot(
    pitches_pca[:, 0:2],
    np.transpose(loadings[0:2, :]),
    targets,
    data.columns,
)

# create percentile range for variance explained graph
var_sum = 0
count = 0
for i in range(len(perc_exp)):
    if var_sum < 0.7:
        var_sum += perc_exp[i]
        count += 1

# create the variacne explanied and PCA heatmap graphs
pitches_df = pd.DataFrame(
    data=pitches_pca,
    columns=[
        "PC1",
        "PC2",
        "PC3",
        "PC4",
        "PC5",
        "PC6",
        "PC7",
        "PC8",
        "PC9",
        "PC10",
        "PC11",
    ],
)
pitches_df = pitches_df.drop(
    columns=["PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10", "PC11"]
)

plt.figure(2)
plt.plot(np.cumsum(per_var))
plt.ylabel("Percentage of Expalined Variance")
plt.xlabel("Principal Component Index")
plt.title("Variance Explained")

plt.figure(3)
plt.title("PCA Heatmap")

ax = sns.heatmap(
    pca.components_,
    cmap="YlGnBu",
    yticklabels=["PCA" + str(x) for x in range(1, pca.n_components_ + 1)],
    xticklabels=list(data.columns),
    cbar_kws={"orientation": "horizontal"},
)
ax.set_aspect("equal")

plt.show()
