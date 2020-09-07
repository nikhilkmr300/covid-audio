#!/bin/bash

# COVID-19 Radiography Database contains images classified as COVID-19, Viral
# Pneumonia and NORMAL. However, for our task, it is relevant only whether the
# images are classified as covid or normal. We treat the Viral Pneumonia class
# as NORMAL. This script generates a new directory data_clean which contains the
# images grouped as covid and normal. normal contains NORMAL and Viral Pneumonia
# from COVID-19 Radiography Database.

rm -rf data_clean

mkdir data_clean
mkdir data_clean/covid
mkdir data_clean/normal

# Using double quotes around $file because of the pesky spaces in the file name.
echo "Copying \"COVID-19 Radiography Database\"/\"COVID-19\" to data_clean/covid..."
for file in ./"COVID-19 Radiography Database"/"COVID-19"/*; do
  cp "$file" data_clean/covid
done

echo "Copying \"COVID-19 Radiography Database\"/\"NORMAL\" to data_clean/normal..."
for file in ./"COVID-19 Radiography Database"/"NORMAL"/*; do
  cp "$file" data_clean/normal
done

echo "Copying \"COVID-19 Radiography Database\"/\"Viral Pneumonia\" to data_clean/normal..."
for file in ./"COVID-19 Radiography Database"/"Viral Pneumonia"/*; do
  cp "$file" data_clean/normal
done
