# APEHR

#model testing:
"python main.py params_mimic_247_d1_p3_d15_1.json"

#only train model
"python main_train.py params_incor_3133_d1_p5_d15_1.json"

#only test model
"python main_test.py params_incor_3133_d1_p5_d15_1.json"


These input files are outputs of script preprocess_mimiciii.py. We provide actual input files for reproduction of our results. So, runing preprocess_mimiciii.py is, initially, optional. DISCLAIMER: we built the data file using the mimic-III dataset; but the file is nothing but a bunch of numbers in binary format; in order to make sense out of the file, one must download mimic-III from https://mimic.physionet.org