#!/bin/bash

# Once you have licensed and got access to the KDD paper data, move it to this
# directory and save it in this directory under the name KDD_paper_data.
# Instructions for obtaining the academic license are at the link below:
# https://covid-19-sounds.org/en/blog/data_sharing.html
# Then unzip all the zip files in KDD_paper_data.

# This script reorganizes the data in KDD_paper_data into a more logical form.

rm -rf data_clean
mkdir data_clean
mkdir data_clean/breath data_clean/cough
mkdir data_clean/breath/asthma
mkdir data_clean/breath/covid
mkdir data_clean/breath/normal
mkdir data_clean/cough/asthma
mkdir data_clean/cough/covid
mkdir data_clean/cough/normal

# The data collected from the Android app and the data collected from the web
# app are organized differently. Need to handle them accordingly.

# Unique ID for each audio sample, in the case where basenames clash.
id=1

# ANDROID
echo "Handling files sourced from Android application..."
# Asthma, reported cough as symptom.
echo -e "\tCopying asthma (cough symptom) breath+cough files to data_clean/*/asthma..."
for subdir in ./KDD_paper_data/asthmaandroidwithcough/*; do
  class=$(basename $subdir)
  # Copying to data_clean/breath/asthma.
  if [ $class == breath ]; then
    for breath_file in $subdir/*; do
      cp $breath_file data_clean/breath/asthma
      filename=$(basename $breath_file)
      mv data_clean/breath/asthma/$filename data_clean/breath/asthma/"BREATH_ASTHMA_withcough_android_[$id]_$filename"
      id=$((id+1))
    done
  # Copying to data_clean/cough/asthma.
  elif [ $class == cough ]; then
    for cough_file in $subdir/*; do
      cp $cough_file data_clean/cough/asthma
      filename=$(basename $cough_file)
      mv data_clean/cough/asthma/$filename data_clean/cough/asthma/"COUGH_ASTHMA_withcough_android_[$id]_$filename"
      id=$((id+1))
    done
  fi
done
# Covid, reported cough as symptom.
echo -e "\tCopying covid (cough symptom) breath+cough files to data_clean/*/covid..."
for subdir in ./KDD_paper_data/covidandroidwithcough/*; do
  class=$(basename $subdir)
  # Copying to data_clean/breath/covid.
  if [ $class == breath ]; then
    for breath_file in $subdir/*; do
      cp $breath_file data_clean/breath/covid
      filename=$(basename $breath_file)
      mv data_clean/breath/covid/$filename data_clean/breath/covid/"BREATH_COVID_withcough_android_[$id]_$filename"
      id=$((id+1))
    done
  # Copying to data_clean/cough/covid.
  elif [ $class == cough ]; then
    for cough_file in $subdir/*; do
      cp $cough_file data_clean/cough/covid
      filename=$(basename $cough_file)
      mv data_clean/cough/covid/$filename data_clean/cough/covid/"COUGH_COVID_withcough_android_[$id]_$filename"
      id=$((id+1))
    done
  fi
done
# Covid, reported no cough.
echo -e "\tCopying covid (no cough symptom) breath+cough files to data_clean/*/covid..."
for subdir in ./KDD_paper_data/covidandroidnocough/*; do
  class=$(basename $subdir)
  # Copying to data_clean/breath/covid.
  if [ $class == breath ]; then
    for breath_file in $subdir/*; do
      cp $breath_file data_clean/breath/covid
      filename=$(basename $breath_file)
      mv data_clean/breath/covid/$filename data_clean/breath/covid/"BREATH_COVID_nocough_android_[$id]_$filename"
      id=$((id+1))
    done
  # Copying to data_clean/cough/covid.
  elif [ $class == cough ]; then
    for cough_file in $subdir/*; do
      cp $cough_file data_clean/cough/covid
      filename=$(basename $cough_file)
      mv data_clean/cough/covid/$filename data_clean/cough/covid/"COUGH_COVID_nocough_android_[$id]_$filename"
      id=$((id+1))
    done
  fi
done
# Normal, reported cough as symptom.
echo -e "\tCopying normal (cough symptom) breath+cough files to data_clean/*/normal..."
for subdir in ./KDD_paper_data/healthyandroidwithcough/*; do
  class=$(basename $subdir)
  # Copying to data_clean/breath/normal.
  if [ $class == breath ]; then
    for breath_file in $subdir/*; do
      cp $breath_file data_clean/breath/normal
      filename=$(basename $breath_file)
      mv data_clean/breath/normal/$filename data_clean/breath/normal/"BREATH_NORMAL_withcough_android_[$id]_$filename"
      id=$((id+1))
    done
  # Copying to data_clean/cough/normal.
  elif [ $class == cough ]; then
    for cough_file in $subdir/*; do
      cp $cough_file data_clean/cough/normal
      filename=$(basename $cough_file)
      mv data_clean/cough/normal/$filename data_clean/cough/normal/"COUGH_NORMAL_withcough_android_[$id]_$filename"
      id=$((id+1))
    done
  fi
done
# Normal, reported no cough.
echo -e "\tCopying normal (no cough symptom) breath+cough files to data_clean/*/normal..."
for subdir in ./KDD_paper_data/healthyandroidnosymp/*; do
  class=$(basename $subdir)
  # Copying to data_clean/breath/normal.
  if [ $class == breath ]; then
    for breath_file in $subdir/*; do
      cp $breath_file data_clean/breath/normal
      filename=$(basename $breath_file)
      mv data_clean/breath/normal/$filename data_clean/breath/normal/"BREATH_NORMAL_nocough_android_[$id]_$filename"
      id=$((id+1))
    done
  # Copying to data_clean/cough/normal.
  elif [ $class == cough ]; then
    for cough_file in $subdir/*; do
      cp $cough_file data_clean/cough/normal
      filename=$(basename $cough_file)
      mv data_clean/cough/normal/$filename data_clean/cough/normal/"COUGH_NORMAL_nocough_android_[$id]_$filename"
      id=$((id+1))
    done
  fi
done

# WEB
echo "Handling files sourced from web application..."
# Asthma, reported cough as symptom.
echo -e "\tCopying asthma (cough symptom) breath+cough files to data_clean/*/asthma..."
for subdir in ./KDD_paper_data/asthmawebwithcough/*; do
  for file in $subdir/*; do
    # Filename contains breath as substring, copying to data_clean/breath/asthma.
    if grep -q breath <<< $file; then
      cp $file data_clean/breath/asthma
      filename=$(basename $file)
      mv data_clean/breath/asthma/$filename data_clean/breath/asthma/"BREATH_ASTHMA_withcough_web_[$id]_$filename"
      id=$((id+1))
    # Filename contains cough as substring, copying to data_clean/cough/asthma.
    elif grep -q cough <<< $file; then
      cp $file data_clean/cough/asthma
      filename=$(basename $file)
      mv data_clean/cough/asthma/$filename data_clean/cough/asthma/"COUGH_ASTHMA_withcough_web_[$id]_$filename"
      id=$((id+1))
    fi
  done
done
# Covid, reported cough as symptom.
echo -e "\tCopying covid (cough symptom) breath+cough files to data_clean/*/covid..."
for subdir in ./KDD_paper_data/covidwebwithcough/*; do
  for file in $subdir/*; do
    # Filename contains breath as substring, copying to data_clean/breath/covid.
    if grep -q breath <<< $file; then
      cp $file data_clean/breath/covid
      filename=$(basename $file)
      mv data_clean/breath/covid/$filename data_clean/breath/covid/"BREATH_COVID_withcough_web_[$id]_$filename"
      id=$((id+1))
    # Filename contains cough as substring, copying to data_clean/cough/covid.
    elif grep -q cough <<< $file; then
      cp $file data_clean/cough/covid
      filename=$(basename $file)
      mv data_clean/cough/covid/$filename data_clean/cough/covid/"COUGH_COVID_withcough_web_[$id]_$filename"
      id=$((id+1))
    fi
  done
done
# Covid, reported no cough as symptom.
echo -e "\tCopying covid (no cough symptom) breath+cough files to data_clean/*/covid..."
for subdir in ./KDD_paper_data/covidwebnocough/*; do
  for file in $subdir/*; do
    # Filename contains breath as substring, copying to data_clean/breath/covid.
    if grep -q breath <<< $file; then
      cp $file data_clean/breath/covid
      filename=$(basename $file)
      mv data_clean/breath/covid/$filename data_clean/breath/covid/"BREATH_COVID_nocough_web_[$id]_$filename"
      id=$((id+1))
    # Filename contains cough as substring, copying to data_clean/cough/covid.
    elif grep -q cough <<< $file; then
      cp $file data_clean/cough/covid
      filename=$(basename $file)
      mv data_clean/cough/covid/$filename data_clean/cough/covid/"COUGH_COVID_nocough_web_[$id]_$filename"
      id=$((id+1))
    fi
  done
done
# Normal, reported cough as symptom.
echo -e "\tCopying normal (cough symptom) breath+cough files to data_clean/*/normal..."
for subdir in ./KDD_paper_data/healthywebwithcough/*; do
  for file in $subdir/*; do
    # Filename contains breath as substring, copying to data_clean/breath/covid.
    if grep -q breath <<< $file; then
      cp $file data_clean/breath/normal
      filename=$(basename $file)
      mv data_clean/breath/normal/$filename data_clean/breath/normal/"BREATH_NORMAL_withcough_web_[$id]_$filename"
      id=$((id+1))
    # Filename contains cough as substring, copying to data_clean/cough/covid.
    elif grep -q cough <<< $file; then
      cp $file data_clean/cough/normal
      filename=$(basename $file)
      mv data_clean/cough/normal/$filename data_clean/cough/normal/"COUGH_NORMAL_withcough_web_[$id]_$filename"
      id=$((id+1))
    fi
  done
done
# Normal, reported no cough as symptom.
echo -e "\tCopying normal (no cough symptom) breath+cough files to data_clean/*/normal..."
for subdir in ./KDD_paper_data/healthywebnosymp/*; do
  # Files in directory 2020-04-07-09_48_27_013250 have no extension. They throw
  # error on generating spectrograms as the software cannot read them as audio
  # files. Not including them in data_clean.
  if [ $(basename $subdir) = 2020-04-07-09_48_27_013250 ]; then
    continue
  fi
  for file in $subdir/*; do
    # Filename contains breath as substring, copying to data_clean/breath/covid.
    if grep -q breath <<< $file; then
      cp $file data_clean/breath/normal
      filename=$(basename $file)
      mv data_clean/breath/normal/$filename data_clean/breath/normal/"BREATH_NORMAL_nocough_web_[$id]_$filename"
      id=$((id+1))
    # Filename contains cough as substring, copying to data_clean/cough/covid.
    elif grep -q cough <<< $file; then
      cp $file data_clean/cough/normal
      filename=$(basename $file)
      mv data_clean/cough/normal/$filename data_clean/cough/normal/"COVID_NORMAL_nocough_web_[$id]_$filename"
      id=$((id+1))
    fi
  done
done
