import h5py
import numpy as np
import glob
import random
import cv2
import os.path as osp

# ---------------------------------------------------------------------
# ----------------------------- functions -----------------------------
# ---------------------------------------------------------------------

def clahe_equalized(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.5,tileGridSize=(8,8))
    lab[...,0] = clahe.apply(lab[...,0])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return bgr

def categorize_diagnosis(path_images):
    """ sorts and categorizes the list of all image paths by their diagosis. Age-related macular degeneration (A),
        glaucoma (G), normal (N) anf diabetic retinopathy (D). They are then shuffled and divided into a big part (80%)
        and a small part (20%).
                        Parameters:
                        ---------------
                        path_images list of all image paths
        """
    # Initialize lists for each category
    files_A = []
    files_G = []
    files_N = []
    files_D = []

    # Function to determine which list to append a file to
    def sort_into_lists(filename):
        if '_A.png' in filename:
            files_A.append(filename)
        elif '_G.png' in filename:
            files_G.append(filename)
        elif '_N.png' in filename:
            files_N.append(filename)
        elif '_D.png' in filename:
            files_D.append(filename)
        else:
            print("Wrong filename: ", filename)

    # Sort files into the respective lists
    for image in path_images:
        sort_into_lists(image)

    # Sort each list
    files_A.sort()
    files_G.sort()
    files_N.sort()
    files_D.sort()

    # Compute indices for big, small (80/20 proportions)
    A_ind_shuffled = list(range(len(files_A)))
    random.shuffle(A_ind_shuffled)
    A_big, A_small = np.split(np.asarray(A_ind_shuffled), [int(len(A_ind_shuffled) * 0.8)])
    A_big = A_big.tolist()
    A_small = A_small.tolist()

    G_ind_shuffled = list(range(len(files_G)))
    random.shuffle(A_ind_shuffled)
    G_big, G_small = np.split(np.asarray(G_ind_shuffled), [int(len(G_ind_shuffled) * 0.8)])
    G_big = G_big.tolist()
    G_small = G_small.tolist()

    N_ind_shuffled = list(range(len(files_N)))
    random.shuffle(N_ind_shuffled)
    N_big, N_small = np.split(np.asarray(N_ind_shuffled), [int(len(N_ind_shuffled) * 0.8)])
    N_big = N_big.tolist()
    N_small = N_small.tolist()

    D_ind_shuffled = list(range(len(files_D)))
    random.shuffle(A_ind_shuffled)
    D_big, D_small = np.split(np.asarray(D_ind_shuffled), [int(len(D_ind_shuffled) * 0.8)])
    D_big = D_big.tolist()
    D_small = D_small.tolist()

    # divide lists
    A_big = [files_A[i] for i in A_big]
    A_small = [files_A[i] for i in A_small]
    G_big = [files_G[i] for i in G_big]
    G_small = [files_G[i] for i in G_small]
    N_big = [files_N[i] for i in N_big]
    N_small = [files_N[i] for i in N_small]
    D_big = [files_D[i] for i in D_big]
    D_small = [files_D[i] for i in D_small]

    return A_big, A_small, G_big, G_small, N_big, N_small, D_big, D_small

def write_range_to_hdf5(counter_from, counter_to, images_data, labels_data, diagnoses_data, ids_data, images,
                                labels, ids, diagnoses):
    """ writes range of 4 images to hdf5 file
                    Parameters:
                    ---------------
                    counter_from   write from
                    counter_to     write to
                    images_data    hdf5 dataset
                    labels_data    hdf5 dataset
                    diagnoses_data hdf5 dataset
                    ids_data       hdf5 dataset
                    images         list of images as numpy arrays
                    labels         list of labels as numpy arrays
                    ids            list of ids as strings
                    diagnosis      list of diagnosis as strings
    """
    # add images
    images = np.asarray(images)
    images_data[counter_from:counter_to] = images

    # add labels
    labels = np.asarray(labels)
    labels_data[counter_from:counter_to] = labels

    # add diagnosis
    dt = h5py.special_dtype(vlen=str)
    diagnoses_data[counter_from:counter_to] = np.asarray(diagnoses, dtype=dt)

    # add ids
    ids_data[counter_from:counter_to] = np.asarray(ids, dtype=dt)


def add_images(image_paths, images_data, labels_data, diagnoses_data, ids_data, counter_from, diagnosis):
    """ preprocesses images and adds them to .
            Parameters:
            ---------------
            image_paths        hdf5 dataset
            images_data        hdf5 dataset
            labels_data        hdf5 dataset
            diagnoses_data     hdf5 dataset
            ids_data           hdf5 dataset
            counter_from       int
            diagnosis          diagnosis (A, G, N, D) string

    """
    max_write_buffer = 4
    write_buffer = 0
    images = []
    labels = []
    ids = []
    diagnoses = []

    # go through all images, then preprocess them and write them to hdf5 files in batches of five
    for i in range(len(image_paths)):
        # write to hdf5 file if write_buffer is full
        if write_buffer >= max_write_buffer:
            # write images to hdf5 file
            counter_to = counter_from + write_buffer
            print("writing ids ", ids, " at indices from", counter_from, " to ", counter_to)
            write_range_to_hdf5(counter_from, counter_to, images_data, labels_data, diagnoses_data, ids_data, images,
                                labels, ids, diagnoses)
            # delete cash/lists
            images.clear()
            labels.clear()
            ids.clear()
            diagnoses.clear()
            # reset stuff for next iteration
            write_buffer = 0
            counter_from += 4

        img_id = osp.splitext(osp.basename(image_paths[i]))[0]
        ids.append(img_id)

        image = cv2.imread(image_paths[i], cv2.IMREAD_COLOR)
        image = clahe_equalized(image)
        image = cv2.resize(image, (320, 320), interpolation=cv2.INTER_NEAREST)
        image = np.asarray(image)
        images.append(image)

        label_path = image_paths[i].replace("Original", "GroundTruth")
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, (320, 320), interpolation=cv2.INTER_NEAREST)
        label = np.asarray(label)
        label = label / 255
        labels.append(label)

        diagnoses.append(diagnosis)

        write_buffer += 1
    # write remaining images to hdf5 if images list still contains images
    if images:
        counter_to = counter_from + write_buffer
        print("writing ids ", ids, " at indices from", counter_from, " to ", counter_to)
        write_range_to_hdf5(counter_from, counter_to, images_data, labels_data, diagnoses_data, ids_data, images,
                            labels, ids, diagnoses)

# -----------------------------------------------------------------------------
# 1. ----------------------------- create H5 file -----------------------------
# -----------------------------------------------------------------------------

h5_file_path = "FIVES.h5"  # output file path
h5_file = h5py.File(h5_file_path, "w")

# ---------------------------------------------------------------------------
# 2. ----------------------------- [train, val] -----------------------------
# ---------------------------------------------------------------------------

path_to_train = "" # path to raw train data

# sort by diagnosis
path_images = glob.glob(path_to_train + "/Original/*.png")
A_train, A_val, G_train, G_val, N_train, N_val, D_train, D_val = categorize_diagnosis(path_images)

# number of images in each set
number_train_images = len(A_train) + len(G_train) + len(N_train) + len(D_train)
number_val_images = len(A_val) + len(G_val) + len(N_val) + len(D_val)

img_data_train = h5_file.create_dataset("train/images", shape=(number_train_images, 320, 320, 3), dtype="uint8")
img_data_val = h5_file.create_dataset("val/images", shape=(number_val_images, 320, 320, 3), dtype="uint8")

label_data_train = h5_file.create_dataset("train/label", shape=(number_train_images, 320, 320), dtype="uint8")
label_data_val = h5_file.create_dataset("val/label", shape=(number_val_images, 320, 320), dtype="uint8")

dt = h5py.special_dtype(vlen=str)
diagnosis_data_train = h5_file.create_dataset("train/diagnosis", shape=(number_train_images), dtype=dt)
diagnosis_data_val = h5_file.create_dataset("val/diagnosis", shape=(number_val_images), dtype=dt)

id_train = h5_file.create_dataset("train/id", shape=(number_train_images), dtype=dt)
id_val = h5_file.create_dataset("val/id", shape=(number_val_images), dtype=dt)

# ---- train ----
add_images(A_train, img_data_train, label_data_train, diagnosis_data_train, id_train, 0, "A")
add_images(G_train, img_data_train, label_data_train, diagnosis_data_train, id_train, len(A_train), "G")
add_images(N_train, img_data_train, label_data_train, diagnosis_data_train, id_train, len(A_train)+len(G_train), "N")
add_images(D_train, img_data_train, label_data_train, diagnosis_data_train, id_train, len(A_train)+len(G_train)+len(N_train), "D")

# ---- val ----
add_images(A_val, img_data_val, label_data_val, diagnosis_data_val, id_val, 0, "A")
add_images(G_val, img_data_val, label_data_val, diagnosis_data_val, id_val, len(A_val), "G")
add_images(N_val, img_data_val, label_data_val, diagnosis_data_val, id_val, len(A_val)+len(G_val), "N")
add_images(D_val, img_data_val, label_data_val, diagnosis_data_val, id_val, len(A_val)+len(G_val)+len(N_val), "D")

# --------------------------------------------------------------------------
# ----------------------------- 3. [cal, test] -----------------------------
# --------------------------------------------------------------------------

path_to_test = ""  # path to raw test data
# sort by diagnosis
path_images = glob.glob(path_to_test + "/Original/*.png")
A_test, A_cal, G_test, G_cal, N_test, N_cal,  D_test, D_cal = categorize_diagnosis(path_images)

# number of images in each set
number_cal_images = len(A_cal) + len(G_cal) + len(N_cal) + len(D_cal)
number_test_images = len(A_test) + len(G_test) + len(N_test) + len(D_test)

img_data_cal = h5_file.create_dataset("cal/images", shape=(number_cal_images, 320, 320, 3), dtype="uint8")
img_data_test = h5_file.create_dataset("test/images", shape=(number_test_images, 320, 320, 3), dtype="uint8")

label_data_cal = h5_file.create_dataset("cal/label", shape=(number_cal_images, 320, 320), dtype="uint8")
label_data_test = h5_file.create_dataset("test/label", shape=(number_test_images, 320, 320), dtype="uint8")

dt = h5py.special_dtype(vlen=str)
diagnosis_data_cal = h5_file.create_dataset("cal/diagnosis", shape=(number_cal_images), dtype=dt)
diagnosis_data_test = h5_file.create_dataset("test/diagnosis", shape=(number_test_images), dtype=dt)

id_cal = h5_file.create_dataset("cal/id", shape=(number_cal_images), dtype=dt)
id_test = h5_file.create_dataset("test/id", shape=(number_test_images), dtype=dt)

# ---- cal ----
add_images(A_cal, img_data_cal, label_data_cal, diagnosis_data_cal, id_cal, 0, "A")
add_images(G_cal, img_data_cal, label_data_cal, diagnosis_data_cal, id_cal, len(A_cal), "G")
add_images(N_cal, img_data_cal, label_data_cal, diagnosis_data_cal, id_cal, len(A_cal)+len(G_cal), "N")
add_images(D_cal, img_data_cal, label_data_cal, diagnosis_data_cal, id_cal, len(A_cal)+len(G_cal)+len(N_cal), "D")

# ---- test ----
add_images(A_test, img_data_test, label_data_test, diagnosis_data_test, id_test, 0, "A")
add_images(G_test, img_data_test, label_data_test, diagnosis_data_test, id_test, len(A_test), "G")
add_images(N_test, img_data_test, label_data_test, diagnosis_data_test, id_test, len(A_test)+len(G_test), "N")
add_images(D_test, img_data_test, label_data_test, diagnosis_data_test, id_test, len(A_test)+len(G_test)+len(N_test), "D")

# --------------------------------------------------------------------------
h5_file.close()