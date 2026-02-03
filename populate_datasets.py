# Import necessary libraries
import tqdm 
from sklearn import datasets
import seaborn as sns
import urllib.request
import numpy as np
import pandas as pd
import zipfile
import tempfile
from glob import glob
from skimage import io, transform
import os

datasets_path = 'datasets'
if not os.path.isdir(datasets_path):
    os.mkdir(datasets_path)
labels_path = "dataset_labels"
if not os.path.isdir(labels_path):
    os.mkdir(labels_path)


def loadEspadatoDatasets():
    """
    Function to download and save datasets from the Espadato website.
    """
    #Grab the website with DR dataset links
    datasetHtml = str(urllib.request.urlopen("https://mespadoto.github.io/proj-quant-eval/post/datasets").read())

    # Split the webpage content to get the block of links for each dataset
    datasetList = datasetHtml.split("</tr>")[1:-1]

    for dataset in tqdm.tqdm(datasetList):
        # Remove header and tail from the third element of the <tr> list to get the dataset name
        header = "<td><a href=\"../../data"
        tail = "\">X.npy</a>"

        qstr = dataset.split("</td>")[3]
        qstr = qstr.replace(header,"https://mespadoto.github.io/proj-quant-eval/data")
        qstr = qstr.replace(tail, "").replace("\\n", "")

        name = qstr.replace("https://mespadoto.github.io/proj-quant-eval/data/", "").replace("/X.npy", "")

        # The ORL dataset is currently not generated properly in the Espadoto code, and does not reflect the actual Olivetti faces dataset, so the resulting analyses are invalid.
        # Therefore, this dataset is skipped in our analysis
        if name == 'orl':
            continue
        
        # The dataset itself
        data = urllib.request.urlopen(qstr)
        # The labels of the dataset
        labels = urllib.request.urlopen(qstr.replace("X.npy", "y.npy"))

        # Write the raw binary data of the dataset to a .npy file
        with open(f'{datasets_path}/{name}.npy', 'wb') as fdata:
            for line in data:
                fdata.write(line)
        
        # Write the label data to a .npy file
        with open(f'{labels_path}/{name}.npy', 'wb') as fdata:
            for line in labels:
                fdata.write(line)

def loadSmallDatasets():
    """
    Function to load smaller, well-known datasets and save them locally.
    """

    # Load iris dataset
    data = datasets.load_iris()
    df = pd.DataFrame(data.data)
    df['target'] = data.target
    df.drop_duplicates(inplace=True)
    X = df[[0, 1, 2, 3]].to_numpy()
    np.save(f"{datasets_path}/iris.npy", X)
    labels = df["target"].to_numpy()
    np.save(f"{labels_path}/iris.npy", labels)

    # Load wine dataset
    data = datasets.load_wine()
    np.save(f"{datasets_path}/wine.npy", data.data)
    np.save(f"{labels_path}/wine.npy", data.target)

    # Load swiss roll dataset
    X, t = datasets.make_swiss_roll(n_samples=1500)
    np.save(f"{datasets_path}/swissroll.npy", X)
    np.save(f"{labels_path}/swissroll.npy", t)

    # Load penguins dataset
    data = sns.load_dataset('penguins').dropna(thresh=6)
    cols_num = ['bill_length_mm', 'bill_depth_mm',
                'flipper_length_mm', 'body_mass_g']
    X = data[cols_num]
    np.save(f"{datasets_path}/penguins.npy", X)
    species_mapping = {species: idx for idx, species in enumerate(data['species'].unique())}
    labels = data['species'].map(species_mapping).to_numpy()
    np.save(f"{labels_path}/penguins.npy", labels)

    # Load auto-mpg dataset
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    data = pd.read_csv(url, sep='\s+', header=None)
    data.columns = ['mpg', 'cylinders', 'displacement', 'horsepower',
                    'weight', 'acceleration', 'model_year', 'origin', 'car_name']
    data.horsepower = pd.to_numeric(data.horsepower, errors='coerce')
    data = data.drop(['model_year', 'origin', 'car_name'], axis=1)
    data = data[data.horsepower.notnull()]
    X = data[['acceleration', 'cylinders',
                'displacement', 'horsepower', 'weight']]
    labels = data['mpg'].to_numpy()
    np.save(f"{datasets_path}/auto-mpg.npy", X)
    np.save(f"{labels_path}/auto-mpg.npy", labels)

    # Load s-curve dataset
    X, t = datasets.make_s_curve(n_samples=1500)
    np.save(f"{datasets_path}/s-curve.npy", X)
    np.save(f"{labels_path}/s-curve.npy", t)

    # Load ORL dataset
    # This dataset exists in the Espadoto Datasets, but is currently loaded incorrectly in the Espadoto repository, and therefore needs to be manually loaded


    url = 'http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip'
    
    # Create a temporary file for the ZIP archive
    with tempfile.NamedTemporaryFile(delete=False) as tmp_zip:
        zip_path = tmp_zip.name
    
    # Download the ZIP file
    urllib.request.urlretrieve(url, zip_path)
    
    try:
        # Create a temporary directory for extraction
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(zip_path, 'r') as orl:
                orl.extractall(tmp_dir)

            subjects = sorted(glob(os.path.join(tmp_dir, 's*')))

            img_h = 112 // 5
            img_w = 92 // 5

            X = np.zeros((len(subjects) * 10, img_h, img_w))
            y = np.zeros((len(subjects) * 10,), dtype='uint8')

            for i, dir_name in enumerate(subjects):
                label = int(os.path.basename(dir_name).replace('s', ''))

                for j in range(10):
                    tmp = io.imread(os.path.join(dir_name, f'{j + 1}.pgm'))
                    tmp = transform.resize(tmp, (img_h, img_w), preserve_range=True)
                    X[10 * i + j] = tmp / 255.0
                    y[10 * i + j] = label

        X = X.reshape((-1, img_h * img_w))

        np.save(f"{datasets_path}/orl.npy", X)
        np.save(f'{labels_path}/orl.npy', y)
    finally:
        # Remove the downloaded ZIP file
        os.remove(zip_path)



if __name__ == "__main__":
    """
    Main function to load and save all datasets.
    """
    loadEspadatoDatasets()
    loadSmallDatasets()