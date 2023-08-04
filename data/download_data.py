import os
import tarfile
import urllib.request
import zipfile


def download_and_unzip(url, save_dir, zipname):
    """Downloads and unzips a (g)zip file from a url.

    Args:
        url (str): The url to download from.
        save_dir (str): The directory to save the (g)zip file to.
        zipname (str): The name of the (g)zip file.

    Returns:
        None
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if len(os.listdir(save_dir)) == 0 or True:
        urllib.request.urlretrieve(url, zipname)
        print("Downloaded data to {}".format(zipname))
        if zipname.split(".")[-1] == "gz":
            with tarfile.open(zipname, "r:gz") as tar:
                tar.extractall(save_dir)
        else:
            with zipfile.ZipFile(zipname, "r") as zip:
                zip.extractall(save_dir)
    else:
        print("Data already exists in {}".format(save_dir))


if __name__ == "__main__":
    data_dir = "data"
    url = (
        "http://www.timeseriesclassification.com/ClassificationDownloads/Archives"
        "/Multivariate2018_arff.zip"
    )
    save_dir = "/data/math-datasig/shug6778/Log-Neural-CDEs/data/raw/UEA/"
    zipname = save_dir + "uea.zip"
    download_and_unzip(url, save_dir, zipname)

    url = "https://storage.googleapis.com/long-range-arena/lra_release.gz"
    save_dir = data_dir + "/raw/LRA/"
    zipname = save_dir + "lra.gz"
    download_and_unzip(url, save_dir, zipname)
