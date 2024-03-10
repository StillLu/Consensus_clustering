
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from matplotlib.patches import Patch
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.manifold import spectral_embedding


def plot_cdf(cm, L, file_path):
    plt.figure(figsize=(10, 6))

    for i, (bin_edges,cdf) in enumerate(cm):

        plt.plot(bin_edges[:-1], cdf, label=f'K={L+i}')

    plt.xlabel('Consensus Index')
    plt.ylabel('CDF')

    plt.title('Area under the CDF for each K')
    plt.legend()
    plt.savefig('./area_under_cdf_plot.png')
    plt.close()


def draw_consus(cs, cl, k, linkage_mat, file_path):

    ### reorder based on the optimal leaf list
    sorted_indices = hierarchy.leaves_list(linkage_mat)
    consensus_matrix = cs[sorted_indices,:][:,sorted_indices]
    cl = cl[sorted_indices]

    fig = plt.figure(figsize=(9, 10))

    matrix_size = 0.6  # Size for both width and height of the heatmap
    dendro_height = 0.15 # Height of the dendrogram
    left_position = 0.1  # Left position for both dendrogram and heatmap
    #colorbar_width = 0.05  # Width of the color bar

    ax_dendro = fig.add_axes([left_position, 1 - dendro_height, matrix_size, dendro_height])
    dendro = hierarchy.dendrogram(linkage_mat, leaf_rotation=90., no_labels=True, ax=ax_dendro,color_threshold=0, above_threshold_color='black')
    ax_dendro.set_xticks([])
    ax_dendro.set_yticks([])

    for spine in ax_dendro.spines.values():
        spine.set_visible(False)

    ax_cmat = fig.add_axes([left_position, 1 - dendro_height - matrix_size, matrix_size, matrix_size])
    cax = ax_cmat.matshow(consensus_matrix, aspect='equal', cmap='Blues')
    ax_cmat.set_xticks([])
    ax_cmat.set_yticks([])

    cl_01 = []
    cl_start = 0
    for i in range(1, len(cl)):
        if cl[i] != cl[cl_start]:
            cl_01.append((cl_start, i))
            cl_start = i
    cl_01.append((cl_start, len(cl)))
    cl_01 = np.array(cl_01)

    ax_cmat.hlines(cl_01[:, 0] + 0.5 - 1, -0.5, len(cl) - 0.5, color='white', linewidth=1)
    ax_cmat.vlines(cl_01[:, 0] + 0.5 - 1, -0.5, len(cl) - 0.5, color='white', linewidth=1)

    clbar_bottom = 1 - dendro_height -0.015

    ax_clbar = fig.add_axes([left_position, clbar_bottom, matrix_size, 0.01])

    cl_data = cl.reshape(1, -1)
    ax_clbar.imshow(cl_data, cmap='tab20c', aspect='auto')
    ax_clbar.axis('off')

    unique_clusters = np.unique(cl)
    n_clusters = len(unique_clusters)

    cmap = plt.get_cmap('tab20c', n_clusters)


    legend_handles = [Patch(color=cmap(i), label=f'Cluster {unique_clusters[i] + 1}') for i in range(n_clusters)]
    ax_clbar.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save or show the figure
    plt.savefig(f"{file_path}/consensus_and_hierarchical_k{k}.png")
    # plt.show()  #


def process_t_data(file_path, variable_names, drop_vals, df_einfo, timepoint):
    df_t = pd.read_csv(file_path, header=None, names=variable_names)
    df_t = df_t.drop(columns=drop_vals, errors='ignore')
    df_t['ID'] = df_t.index + 1
    df_t = df_t[df_t['ID'].isin(df_einfo['ID'])]
    df_t['timepoint'] = timepoint
    return df_t

def process_rep_data(file_path, df_einfo, df_einfo_train, df_einfo_test, timepoint):
    rep_data_all = pd.read_csv(file_path, header=None)
    rep_data_all['ID'] = rep_data_all.index + 1
    rep_data_all['timepoint'] = timepoint
    rep_data_all = rep_data_all[rep_data_all['ID'].isin(df_einfo['ID'])]

    rep_data = rep_data_all[rep_data_all['ID'].isin(df_einfo_train['ID'])]
    rep_data_val = rep_data_all[rep_data_all['ID'].isin(df_einfo_test['ID'])]

    return rep_data_all, rep_data, rep_data_val


def process_risk_data(file_path, column_name, df_einfo, df_einfo_train, df_einfo_test):
    risk_data_all = pd.read_csv(file_path, names=[column_name])
    risk_data_all['ID'] = risk_data_all.index + 1
    risk_data_all = risk_data_all[risk_data_all['ID'].isin(df_einfo['ID'])]

    risk_data = risk_data_all[risk_data_all['ID'].isin(df_einfo_train['ID'])]
    risk_data_val = risk_data_all[risk_data_all['ID'].isin(df_einfo_test['ID'])]

    return risk_data_all, risk_data, risk_data_val



def ConvertToDM(data):
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    if data.index.name != 'ID':
        data = data.set_index('ID')

    res = data.values

    return res



def derive_extra_features(df):

    for i in range(1, 5):
        df[f'to_{i}'] = np.where(df['cluster6'] == i, 1, 0)
        df[f'in_{i}'] = np.where(df['cluster0'] == i, 1, 0)


    df['time_2_abx'] = df['time_2_abx'].fillna(np.inf)
    df['age_18_55'] = np.where((df['Age'] >= 18) & (df['Age'] <= 55), 1, 0)
    df['age_55_67'] = np.where((df['Age'] > 55) & (df['Age'] < 67), 1, 0)
    df['age_67'] = np.where(df['Age'] >= 67, 1, 0)


    df['SOFA_3_6_cf'] = df['SOFA_3_6'].fillna(df['SOFA_0_3'])
    df['sofa_change'] = df['SOFA_3_6'] - df['SOFA_0_3']
    df['sofa_increase'] = np.where(df['sofa_change'] > 0, 1, 0)


    df['abx_given'] = np.where((df['time_2_abx'] <= 6) & (df['time_2_abx'] >= 3), 1, 0)

    df['volumn_fluids_in_0_3'] = df['fluids_in_0_3'].fillna(0) / 1000
    df['volumn_fluids_in_3hr_of_sepsis'] = df['fluids_sepsis_to_T3'].fillna(0) / 1000
    df['volumn_fluids_in_6hr_of_sepsis'] = df['fluids_sepsis_to_T6'].fillna(0) / 1000

    df['fluids_ge_30mlkg'] = np.where(df['volumn_fluids_in_0_3'] >= 0.03 * df['weight_in_kg'], 1, 0)
    df['fluids_gt_15mlkg_lt_30mlkg'] = np.where((df['volumn_fluids_in_0_3'] < 0.03 * df['weight_in_kg']) &
                                                (df['volumn_fluids_in_0_3'] > 0.015 * df['weight_in_kg']), 1, 0)
    df['fluids_le_15mlkg'] = np.where((df['volumn_fluids_in_0_3'] < 0.015 * df['weight_in_kg']) &
                                      (df['volumn_fluids_in_0_3'] != 0), 1, 0)
    df['fluids_le_30mlkg'] = np.where((df['volumn_fluids_in_0_3'] < 0.03 * df['weight_in_kg']) &
                                      (df['fluids_in_3_hours'] != 0), 1, 0)
    df['log_volume_fluids_in_0_3'] = np.log(df['fluids_in_0_3'].replace(0, np.nan) / 1000)
    df['fluids_in_0_6'] = df['fluids_in_0_6'].fillna(0) / 1000


    for col in ['wscore', 'chf', 'mld', 'msld', 'rend']:
        df[col] = df[col].fillna(0)

    return df



def reorder_linkage_gw(
    linkage: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    '''reorder_linkage_gw

    Reorder linkage matrix using the algorithm described by Gruvaeus & Wainer (1972) [1]_.

    Parameters
    ----------
    linkage : numpy.ndarray
        Linkage matrix as returned from scipy.cluster.hierarchy.linkage.
    dist : numpy.ndarray
        n * n distance matrix.

    Returns
    -------
    numpy.ndarray
        Reordered linkage matrix.

    References
    ----------
    .. [1]  Gruvaeus, G., H., Wainer. 1972. Two Additions to Hierarchical Cluster Analysis.
            The British Psychological Society 25.
    '''
    linkage = linkage.copy()

    n = linkage.shape[0]

    # left and right leaves of a cluster
    l_r = np.zeros((n, 2))
    # matrix determining, whether a cluster (subtree) should be flipped
    flip = np.full((n, 2), False)

    # find left and right leaves of clusters
    # and determine, whether cluster should
    # be flipped
    for i in range(n):
        l, r = linkage[i, [0, 1]].astype(int)

        # l and r are singletons
        if l <= n and r <= n:
            l_r[i] = (l, r)
        # only l is a singleton
        elif l <= n:
            l_r[i, 0] = l

            # left and right leaves of cluster r
            rl, rr = l_r[r - (n + 1)].astype(int)

            if dist[l, rl] < dist[l, rr]:
                l_r[i, 1] = rr
            else:
                l_r[i, 1] = rl
                flip[i, 1] = True
        # only r is singleton
        elif r <= n:
            l_r[i, 1] = r

            # left and right leaves of cluster l
            ll, lr = l_r[l - (n + 1)].astype(int)

            if dist[r, ll] < dist[r, lr]:
                l_r[i, 0] = lr
                flip[i, 0] = True
            else:
                l_r[i, 0] = ll
        # none of l and r are singletons
        else:
            # left and right leaves
            ll, lr = l_r[l - (n + 1)].astype(int)
            rl, rr = l_r[r - (n + 1)].astype(int)

            d_ll_rl = dist[ll, rl] # 0
            d_ll_rr = dist[ll, rr] # 1
            d_lr_rl = dist[lr, rl] # 2
            d_lr_rr = dist[lr, rr] # 3

            mn_idx = np.argmin([d_ll_rl, d_ll_rr, d_lr_rl, d_lr_rr])
            if mn_idx == 0: # d_ll_rl
                l_r[i] = (lr, rr)
                flip[i, 0] = True
            elif mn_idx == 1: # d_ll_rr
                l_r[i] = (lr, rl)
                flip[i] = (True, True)
            elif mn_idx == 2: # d_lr_rl
                l_r[i] = (ll, rr)
            else: # d_lr_rr
                l_r[i] = (ll, rl)
                flip[i, 1] = True

    # apply flip
    for i in range((n-1), 0, -1):
        if flip[i, 0]:
            c = linkage[i, 0].astype(int)
            # non-singleton cluster
            if c > n:
                c = c - (n + 1)
                linkage[c, [0, 1]] = linkage[c, [1, 0]]
                if flip[c, 0] == flip[c, 1]:
                    flip[c] = ~flip[c]
        if flip[i, 1]:
            c = linkage[i, 1].astype(int)
            if c > n:
                c = c - (n + 1)
                linkage[c, [0, 1]] = linkage[c, [1, 0]]
                if flip[c, 0] == flip[c, 1]:
                    flip[c] = ~flip[c]

    return linkage
