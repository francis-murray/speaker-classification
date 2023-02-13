import glob
import math

import numpy as np
import seaborn as sns
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from python_speech_features import mfcc
from tqdm.notebook import tqdm


def extract_mfcc_features(speaker_id, numcep, speaker_folder):
    """
    Extract the MFCC features for the designated speaker
    
    Args:
        speaker_id: integer - id of the speaker for which to extract MFCCs
    Returns:
        speaker_mfcc: numpy array of shape (nb audio files, numcep) 
                      containing 1 mfcc row vector per audio file of the speaker
    """
    speaker_mfcc = np.zeros((0, numcep))  # create new array for mfcc features for this speaker
    speaker_audio_file_count = 0

    speaker_path = speaker_folder + str(speaker_id)
    print(f"Speaker {speaker_id:>4}:", end=" ")

    # iterate through all the audio files for the designated speaker
    for path in glob.glob(f'{speaker_path}/*/*.flac', recursive=True):
        with open(path, 'rb') as f:
            data, samplerate = sf.read(f)

            # extract MFCC features for the current audio file
        mfcc_features = mfcc(data, samplerate, numcep=numcep, nfilt=numcep)

        # average the MFCC features for the current audio file
        mfcc_features_mean = np.expand_dims(np.mean(mfcc_features, axis=0), axis=1).T

        # add the mean MFCC vector as a new row to the speaker MFCC 2d array
        speaker_mfcc = np.vstack((speaker_mfcc, mfcc_features_mean))

        speaker_audio_file_count += 1

    print(f"{speaker_audio_file_count:>3} audio files", end=", ")

    return speaker_mfcc


def build_dataset(speaker_ids, speakers_gender, numcep, speaker_folder):
    """
    Build dataset for list of speakers passed in as argument
    
    Args:
        speaker_ids: list of ids of the speakers for which to extract MFCCs
        speakers_gender: dictionary mapping speaker ids to their respective gender
        numcep: integer representing the number of cepstrum to extract from each audio file
    Returns:
        dataset: numpy array of shape (nb_speakers * nb_audio_files_per_speaker, numcep + gender_label) 
                (1 mfcc row vector per audio file for each speaker)
    """

    dataset = np.zeros((0, numcep + 1))  # dataset is comprised of numcep features (MFCCs) and 1 label

    # iterate over all speakers
    for i, speaker_id in enumerate(tqdm(speaker_ids, desc='Extracting MFCCs')):
        # extract MFCC features for 1 speaker
        speaker_mfcc = extract_mfcc_features(speaker_id, numcep, speaker_folder)

        # gender column to be added for this user
        gender = speakers_gender[int(speaker_id)]
        print(f"gender {gender} {'(M)' if gender == 0 else '(F)'}")

        gender_col = np.empty(speaker_mfcc.shape[0])
        gender_col.fill(gender)

        # Adding gender column to 2D NumPy Array
        speaker_mfcc_gender = np.column_stack((speaker_mfcc, gender_col))

        # add this user to all speakers
        dataset = np.vstack((dataset, speaker_mfcc_gender))

    return dataset


def split_speakers(speaker_ids, ratio, seed=1):
    """
    Split speaker ids according to split ratio
    
    Args:
        speaker_ids: list of ids of the speakers for which to extract MFCCs
        ratio: float percentage of training set
        seed: integer for reproducibility
    Returns:
        speakers_ids_tr, speakers_ids_te: 2 lists of train and test speaker ids respectively
    """
    np.random.seed(seed)  # set seed

    # generate random indices
    indices = np.random.permutation(len(speaker_ids))

    # calculate number of samples in training set
    nb_tr_samples = int(np.floor(ratio * len(speaker_ids)))

    # split the indices between training an testing set
    ind_tr = indices[:nb_tr_samples]
    ind_te = indices[nb_tr_samples:]

    # get ids corresponding to indices
    speakers_ids_tr = [speaker_ids[i] for i in ind_tr]
    speakers_ids_te = [speaker_ids[i] for i in ind_te]

    return speakers_ids_tr, speakers_ids_te


def show_hist_box(data, attr_list, title, img_path, do_save_to_disk, num_plots):
    """
    Plots aligned histograms and boxplots.
    Args:
        data: shape=(N,D) data array
        attr_list: shape=(D,) list of strings
        title: the title of the plot
        img_path: the path to save the plot
        do_save_to_disk: boolean, whether to save the plot to disk
    """
    fig = plt.figure(figsize=(18, 30))
    outer = gridspec.GridSpec(math.ceil(num_plots / 3), 3, wspace=0.2, hspace=0.5, top=0.950, bottom=0.03)

    for i, attribute in enumerate(attr_list):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[i], hspace=0.1, height_ratios=[1, 4]
        )

        # Boxplot
        ax0 = plt.Subplot(fig, inner[0])
        sns.boxplot(x=data[:, i], ax=ax0)
        fig.add_subplot(ax0)
        ax0.set(xlabel="")
        ax0.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,  # labels along the bottom edge are off
        )

        # Histogram
        ax1 = plt.Subplot(fig, inner[1])
        sns.histplot(data[:, i], bins=50, ax=ax1)
        fig.add_subplot(ax1)
        ax1.set_xlabel(f"{attr_list[i]}")

    # Set title
    fig.suptitle(f"{title}\n({len(data):,} frames)", fontweight="bold")

    # Save to disk
    if do_save_to_disk == True:
        filename = (
                "histplot_"
                + str(title).lower().replace(" ", "_").replace("$", "").replace("'", "")
                + ".jpg"
        )
        fig.savefig(img_path + filename, dpi=300, bbox_inches="tight")

    plt.show()


def remove_outliers(X_train, y_train, iqr_factor):
    Q1, Q3 = np.percentile(X_train, [25, 75], axis=0)
    IQR = Q3 - Q1
    lower_bounds = Q1 - iqr_factor * IQR
    upper_bounds = Q3 + iqr_factor * IQR

    with np.printoptions(precision=2, threshold=10):
        print(f"• Q1: {Q1}")
        print(f"• Q3: {Q3}")
        print(f"• IQR:{IQR}")
        print(f"• lower_bounds (Q1 - {iqr_factor} * IQR): {lower_bounds}")
        print(f"• upper_bounds (Q3 + {iqr_factor} * IQR): {upper_bounds}")

    # find indexes of all rows above lower bounds
    rows_above_lower_bounds_idx = np.all(np.greater(X_train, lower_bounds), axis=1)

    # find indexes of all rows below upper bounds
    rows_below_upper_bounds_idx = np.all(np.less(X_train, upper_bounds), axis=1)

    # find indexes of all rows inside bounds
    rows_inside_bounds = np.logical_and(rows_above_lower_bounds_idx, rows_below_upper_bounds_idx)

    # keep only rows that have values above lower bounds
    X_train_no_outliers = X_train[rows_inside_bounds, :]
    y_train_no_outliers = y_train[rows_inside_bounds, :]

    count_removed = len(X_train) - len(X_train_no_outliers)
    print(f"\nRemoved {count_removed:,} outliers ({count_removed / len(X_train):.1%})")

    return X_train_no_outliers, y_train_no_outliers


def show_confusion_matrix(cf_matrix, title):
    """
    Plot confusion matrix with count and percentage, and show statistics in text box
    Args:
        cf_matrix: shape=(2,2), the confusion matrix
        title: the title of the plot
    Returns:
        fig: the figure
    """

    ######################### Confusion matrix heatmap ###################################
    # adapted from https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.2))
    outcome_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    outcome_counts = ["{:,}".format(value) for value in cf_matrix.flatten()]
    outcome_percentages = [
        "({0:.2%})".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)
    ]
    labels = [
        f"{v1}\n{v2}\n{v3}"
        for v1, v2, v3 in zip(outcome_names, outcome_counts, outcome_percentages)
    ]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(
        cf_matrix,
        annot=labels,
        ax=ax,
        fmt="",
        cmap="Blues",
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_xlabel(f"Predicted label")
    ax.set_ylabel(f"True label")

    # invert axes to have True Positives on top left
    ax.invert_xaxis()
    ax.invert_yaxis()

    fig.suptitle(
        f"{title}\n({cf_matrix.sum():,} observations)\n",
        fontweight="bold",
        fontsize=9.5,
    )

    ############################### Statistics #########################################
    accuracy, precision, recall, f1_score = compute_statistics(cf_matrix)

    stats_text = "Accuracy = {:0.3f}\nPrecision = {:0.3f}\nRecall = {:0.3f}\nF1 Score = {:0.3f}".format(
        accuracy, precision, recall, f1_score
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.3)
    ax.text(
        1.5,
        0.5,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="center",
        bbox=props,
    )
    fig.tight_layout(pad=0.3, h_pad=0, w_pad=0)
    plt.show()
    return fig


def compute_statistics(cf_matrix):
    """
    Computes the statistics of a confusion matrix.
    Args:
        cf_matrix: shape=(2,2), the confusion matrix
    Returns:
        accuracy: the accuracy according to the confusion matrix
        precision: the precision according to the confusion matrix
        recall: the recall according to the confusion matrix
        f1: the f1 score according to the confusion matrix
    """
    accuracy = np.trace(cf_matrix) / float(np.sum(cf_matrix))
    precision = cf_matrix[1, 1] / sum(cf_matrix[:, 1])  # tp / tp + fp
    recall = cf_matrix[1, 1] / sum(cf_matrix[1, :])  # tp / tp + fn
    f1_score = 2 * precision * recall / (precision + recall)

    return accuracy, precision, recall, f1_score
