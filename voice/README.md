# voice
This is the sub-directory for the approach using x-ray images.

## Instructions
The data has not been uploaded to this repository.
It has been acquired through an academic license from the University of Cambridge.
Refer [this](https://covid-19-sounds.org/en/blog/data_sharing.html) link for instructions on licensing their data.

Once you have downloaded the dataset, move it to the `./data` directory. Then unzip all the zip files in the directory.

To understand how to generate the spectrograms, skip to the Generating the spectrograms subsection below.

### Understanding the data
Run `make` in `./data` to preprocess the data.
This will generate several other directories with the data in a more organized form as required by this project.  
* `data_clean`: Raw data in a more organized form.  
* `data_<AUDIOTYPE>`: Data grouped by type of audio (breath, cough) and split into train, validation and test sets.

Files are renamed in `data_clean` according to the convention `AUDIOTYPE_CLASS_isCoughSymptom_datasource_[uniqueID]_originalFileName`.  
* AUDIOTYPE: Type of audio data -- breath, cough.  
* CLASS: Class for the classification model -- asthma, covid, normal.  
* isCoughSymptom: Whether the patient had cough as a symptom or not.  
* datasource: Source of the data -- Android application, web application. 
* originalFileName: Original name of the file as given in the raw data.  
* uniqueID: Several files collected from the web application have the same names. Unique ID is to prevent overwriting on copying to `data_clean`.  

### Generating the spectrograms
To generate the log spectrograms, run `make` in `./data_spec/log_spec`. Similarly to generate the mel spectrograms, run `make` in `./data_spec/mel_spec`. 

It uses the same train-valid-test split used in the `./data` directory. You need not have run `make` in `./data` and in case you haven't, it will automatically organize the data for you before generating the spectrograms. This can take some time.

#### Exploratory data analysis
The sub-directory `./eda` contains some exploratory data analysis on the training audio samples. The pickle files contain data on the distributions of features among the training audio samples, grouped by class (asthma, covid, normal).

#### Feature extraction
Two-fold approach:  
* Classical machine learning models using handcrafted audio features.
* Neural network models using spectrogram images generated from the audio samples.

The `feature_extraction` sub-directory deals with the first approach, i.e., extracting handcrafted audio features.

Audio features come in two types:  
* Instantaneous: Extracted for each frame in an audio sample.
* Global: Extracted for the audio sample as a whole.
To pass the features into our model, we need representations of the instantaneous features (calculated over each frame) over the whole audio sample. So we use aggregate statistics to summarize the instantaneous features over the whole audio sample. You can change which features to use and the statistics used to aggregate the instantaneous features in `./feature_extraction/generate_features.py`.

Run `make` in `./feature_extraction` to generate the train, valid and test CSV files for each audio type. It uses the same train-valid-test split used in the `./data` directory. Again, you need to have run `make` in `./data` explicitly, the scripts take care of the split automatically.
