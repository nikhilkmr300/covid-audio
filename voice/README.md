# voice
This is the sub-directory for the approach using x-ray images.

## Instructions
The data has not been uploaded to this repository.
It has been acquired through an academic license from the University of Cambridge.
Refer [this](https://covid-19-sounds.org/en/blog/data_sharing.html) link for instructions on licensing their data.

Once you have downloaded the dataset, move it to the `./data` directory. Then unzip all the zip files in the directory.

Run `make` in `./data` to preprocess the data.
This will generate several other directories with the data in a more organized form as required by this project.
* `data_clean`: Raw data in a more organized form.
* `data_<AUDIOTYPE>`: Data grouped by type of audio (breath, cough) and split into train, validation and test sets.

Files are renamed in `data_clean` according to the convention `AUDIOTYPE_CLASS_isCoughSymptom_datasource_originalFileName_[uniqueID]`.  
* AUDIOTYPE: Type of audio data -- breath, cough.  
* CLASS: Class for the classification model -- asthma, covid, normal.  
* isCoughSymptom: Whether the patient had cough as a symptom or not.  
* datasource: Source of the data -- Android application, web application. 
* originalFileName: Original name of the file as given in the raw data.  
* uniqueID: Several files collected from the web application have the same names. Unique ID is to prevent overwriting on copying to `data_clean`.  
