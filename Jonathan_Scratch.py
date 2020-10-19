from gensim.models import LdaModel
from gensim.corpora import Dictionary

import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering


def main():

    # Load a pretrained model from disk.
    lda = LdaModel.load("RC_LDA_50_True.lda")

    # Load in the dictionary
    dictionary = ["Tobacco vs Marijuana vs Alcohol", "UNCLEAR/JUNK", "Growing Weed", "Liberal vs Conservative",
                  "Quantity/Duration", "UNCLEAR/JUNK", "Government + Systemic Power", "War on Drugs",
                  "Federal vs State vs Local Government", "Information on Marijuana", "Crime", "Legal Marijuana Market",
                  "Marijuana as it Relates to Other Drugs", "Legalizing Marijuana", "Existing Marijuana Legalization",
                  "Marijuana Busting", "Medicinal Marijuana", "Marijuana Plant Contents", "International + Employment",
                  "FDA Scheduled Drugs", "UNCLEAR/JUNK", "President and International Affairs", "Family Abuse",
                  "Canada and Housing Prices", "Smoking Marijuana", "Private Industry", "West Coast + Supplements",
                  "International", "Police", "Legal Case", "UNCLEAR/JUNK", "Reddit", "UNCLEAR/JUNK", "UNCLEAR/JUNK",
                  "Legal", "Drug Test", "UNCLEAR/JUNK", "Criminal Punishment", "Legalizing Marijuana", "Local delivery",
                  "Marijuana Regulation", "Marijuana Research", "Driving Drunk/High", "Racial Disparities",
                  "Court Case", "Smoking Marijuana/Tobacco/E-Cig", "Edible Foods/Mixed", "Law Enforcement",
                  "Gun Control", "Expletives"]

    # Get the distribution
    distribution = lda.get_topics()

    # Remove junk indices [1,5,18,20,21,23,26,27,30,31,32,33,36,46]
    remove_index = np.array([1, 5, 18, 20, 21, 23, 26, 27, 30, 31, 32, 33, 36, 46])

    # track array
    track_array = np.arange(50)

    # Remove the junk indices
    for i in range(len(remove_index)):
        current_index = remove_index[len(remove_index) - i - 1]
        distribution = np.delete(distribution, current_index, 0)
        track_array = np.delete(track_array, current_index, 0)

    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity="l1", linkage="complete")
    model.fit(distribution)
    plt.title('Hierarchical Clustering Dendrogram')

    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode='level', p=100, leaf_label_func=(lambda id: dictionary[track_array[id]]),
                    leaf_rotation=90)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.tight_layout()
    plt.show()


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        # counts[i] = dictionary[track_array[current_count]]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


if __name__ == "__main__":
    main()
