import os
import urllib.request
import zipfile


def download_and_unzip(url, save_dir, zipname):
    """Downloads and unzips a zip file from a url.

    Args:
        url (str): The url to download from.
        save_dir (str): The directory to save the zip file to.
        zipname (str): The name of the zip file.

    Returns:
        None
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if len(os.listdir(save_dir)) == 0:
        urllib.request.urlretrieve(url, zipname)
        print("Downloaded UEA data to {}".format(zipname))
        with zipfile.ZipFile(zipname, "r") as zip_ref:
            zip_ref.extractall(save_dir)


if __name__ == "__main__":
    url = (
        "http://www.timeseriesclassification.com/Downloads/Archives"
        "/Univariate2018_arff.zip"
    )
    save_dir = "data/UEA"
    zipname = "data/UEA/Univariate2018_arff.zip"
    download_and_unzip(url, save_dir, zipname)
