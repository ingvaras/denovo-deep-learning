# denovo-deep-learning

This is the code used to train and evaluate machine learning models for the master
thesis 'Deep Learning
Architecture Comparison for the Detection of <i>De Novo</i> Mutations in Next Generation Sequencing Data'. The purpose
of it was to explore various deep learning architectures in order to find a potential improvement to DeNovoCNN [1] model
used for mutations detection in sequencing data.

## Setup

The data used for training and evaluation of <i>de novo</i> deep learning models can be found in
this <a href="https://github.com/Genome-Bioinformatics-RadboudUMC/DeNovoCNN_training_dataset">DeNovoCNN GitHub
repository</a> [2].

To download the data and extract it run the following commands in the project directory:

```
pip install gdown
gdown 'https://drive.google.com/uc?id=1FwQt6jq9f2YG1X2lpmgV_oyfv9z6Yf9p'
tar -xf training_dataset.tar.gz
mkdir data
mv publish_images/* data/
rm -r publish_images
rm training_dataset.tar.gz
```

## Results

| Model           | Recall | Precision | Accuracy | F1Score |
|-----------------|--------|-----------|----------|---------|
| DeNovoEnsemble  |        |           |          |         |
| DeNovoResNet    | 0.9789 | 0.9623    | 0.9907   | 0.9705  |
| DeNovoDenseNet  | 0.9731 | 0.9651    | 0.9903   | 0.9691	 |
| DeNovoViT       |        |           |          |         |
| DeNovoInception |        |           |          |         |
| DeNovoCNN [1]   | 0.9674 | 0.9655    | 0.9895   | 0.9665  |

| Model           | True positives | False positives | True negatives | False negatives |
|-----------------|----------------|-----------------|----------------|-----------------|
| DeNovoEnsemble  |                |                 |                |                 |
| DeNovoResNet    | 1531           | 60              | 8355           | 33              |
| DeNovoDenseNet  | 1522           | 55              | 8355           | 42              |
| DeNovoViT       |                |                 |                |                 |
| DeNovoInception |                |                 |                |                 |
| DeNovoCNN [1]   | 1513           | 54              | 8350           | 51              |

## References

[1] Khazeeva, G., Sablauskas, K., van der Sanden, B., Steyaert, W., Kwint, M., Rots, D., Hinne, M., van Gerven, M.,
Yntema, H., Vissers, L., & Gilissen, C. (2022) 'DeNovoCNN: A Deep Learning Approach to De Novo Variant Calling in Next
Generation Sequencing Data', Nucleic Acids Research, 50(17) [online]. Available
at: https://doi.org/10.1093/nar/gkac511 (Accessed: 19 February 2024).

[2] Sablauskas, K., & Khazeeva, G. (2022) ‘DeNovoCNN training dataset’, Genome-Bioinformatics-RadboudUMC GitHub, commit
6662a19 [online]. Available at: https://github.com/Genome-Bioinformatics-RadboudUMC/DeNovoCNN_training_dataset (
Accessed: 25 February 2024).
