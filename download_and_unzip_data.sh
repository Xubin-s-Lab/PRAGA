#!/bin/bash

# Download data package
wget -O Data.zip "https://zenodo.org/records/14591305/files/Data.zip?download=1"

# Check if download was successful
if [ $? -ne 0 ]; then
    echo "Download failed. Please check your network connection or URL."
    exit 1
fi

# Unzip data
unzip Data.zip

# Check if unzip was successful
if [ $? -ne 0 ]; then
    echo "Unzip failed. Please check if Data.zip file is complete."
    exit 1
fi

echo "Data download and unzip completed."