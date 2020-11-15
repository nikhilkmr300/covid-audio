import librosa
import os
import sys
import pickle

sys.path.append('..')
sys.path.append(os.path.join('..', 'data_struc'))

from set_audio_params import *
from load_utils import *
from generate_features import *

def predict(filepath, classifier, audio_type, is_cough_symptom):
    """
    Predicts whether an audio sample is normal or covid.

    Parameters:
    filepath (str): Path to audio file.
    classifier (sklearn.base.BaseEstimator): scikit-learn estimator object with predict method.
    audio_type (str): Can be 'breath' or 'cough'.
    is_cough_symptom (boolean): Corresponds to is_cough_symptom in train.csv, whether the user reported cough as a symptom or not.

    Returns:
    int: Output of predict method of classifier (by default 0 for normal, 1 for covid).
    """

    row_df = extract_features(filepath, audio_type)
    row_df['is_cough_symptom'] = is_cough_symptom

    row_df.to_csv('tmp.csv')

    return model.predict(row_df)

def extract_features(filepath, audio_type):
    """
    Extracts the same features as used to train the model (refer set_audio_params.py or data_struc/train.csv) from the input audio sample at filepath.

    Wrapper around generate_feature_row from generate_features.py.

    Parameters:
    filepath (str): Path to audio file.
    audio_type (str): Can be 'breath' or 'cough'.

    Returns:
    pandas.DataFrame: Row of features as a Pandas DataFrame.
    """

    features = generate_feature_names()

    # No target column as we need to predict.
    orig_df = pd.DataFrame(columns=features + ['filename'])
    orig_df = orig_df.set_index('filename')

    if audio_type == 'breath':
        waveform, _ = load(filepath, sampling_rate, time_per_sample_breath)
    elif audio_type == 'cough':
        waveform, _ = load(filepath, sampling_rate, time_per_sample_cough)

    # class_ parameter is used only if feature_name is 'target'. However, we don't pass 'target' as the name of a feature column, so passing a dummy value for class_.
    filename = os.path.basename(filepath)
    row_df = generate_feature_row(orig_df, filename, -1, waveform, sampling_rate, samples_per_frame, hop_length)

    return row_df

if __name__ == '__main__':
    with open('model_ML_cough.pkl', 'rb') as f:
        model = pickle.load(f)

    filepath = os.path.join('..', 'data_raw', 'data_breath', 'test', 'covid', 'BREATH_COVID_withcough_web_[996]_audio_file_breathe.wav')
    audio_type = 'breath'
    is_cough_symptom = 1
    print(predict(filepath, model, audio_type, is_cough_symptom))