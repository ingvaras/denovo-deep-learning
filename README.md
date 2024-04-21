# denovo-deep-learning

The data used for training and evaluation of <i>de novo</i> deep learning models can be found in
this <a href="https://github.com/Genome-Bioinformatics-RadboudUMC/DeNovoCNN_training_dataset">GitHub repository</a> (
Sablauskas and Khazeeva, 2022).

To download the data and extract it run the following commands in the terminal:

```
pip install gdown
gdown 'https://drive.google.com/uc?id=1FwQt6jq9f2YG1X2lpmgV_oyfv9z6Yf9p'
tar -xf training_dataset.tar.gz
mkdir data
mv publish_images/* data/
rm -r publish_images
rm training_dataset.tar.gz
```

For Mac M1 Pro chip compatibility with tensorflow training on GPU, following tensorflow versions were used:

```
tensorflow-macos==2.15.0
tensorflow-metal==1.1.0
```