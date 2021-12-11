# create a directory
cd ..
mkdir -pv dataset
mkdir -pv dataset/brain_tumor_dataset
cd dataset/brain_tumor_dataset

# download the dataset
wget -O temp.zip -q https://ndownloader.figshare.com/articles/1512427/versions/5 --show-progress

# unzip the dataset and delete the zip
unzip -q temp.zip && rm temp.zip

# concatenate the multiple zipped data in a single zip
cat brainTumorDataPublic_* > brainTumorDataPublic_temp.zip
zip -FF brainTumorDataPublic_temp.zip --out data.zip > progress.txt

# remove the temporary files
rm brainTumorDataPublic_*

# unzip the full archive and delete it
unzip -q data.zip -d data && rm data.zip

# check that "data" contains 3064 files
ls data | wc -l

cd ../..

python file_conversion/mat_to_nifti.py "/media/fabio/Disco locale1/Fabio/Programmazione/Python/segmentation/dataset/brain_tumor_dataset/"