import zipfile

z = zipfile.ZipFile("/home/prinjoh/Kaggle/train.csv.zip")
z.extractall("/home/prinjoh/Kaggle")