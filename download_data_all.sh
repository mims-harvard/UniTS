#!/bin/bash

mkdir -p dataset

# check for gdown https://github.com/wkentaro/gdown then install if necessary
if ! command -v gdown &> /dev/null
then
    echo "installing gdown, for downloading from google drive"
    pip install gdown
fi

# TimesNet data
# downloads all_datasets.zip and extracts into dataset/
if [ ! -f dataset/all_datasets.zip ]; then
    gdown "https://drive.google.com/file/d/1pmXvqWsfUeXWCMz5fqsP8WLKXR5jxY8z/view?usp=drive_link" --fuzzy -O dataset/all_datasets.zip
    unzip dataset/all_datasets.zip -d dataset/
    mv dataset/all_datasets/* dataset/
    rm -rf dataset/all_datasets
fi

# UAE data
# downloads Multivariate2018_ts.zip then extacts into dataset/UAE/
if [ ! -f dataset/Multivariate2018_ts.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip" -O dataset/Multivariate2018_ts.zip
    unzip dataset/Multivariate2018_ts.zip -d dataset/
    mv dataset/Multivariate_ts dataset/UAE
fi

# UCR data
# downloads Univariate2018_ts.zip then extacts into dataset/UCR/
if [ ! -f dataset/Univariate2018_ts.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip" -O dataset/Univariate2018_ts.zip
    unzip dataset/Univariate2018_ts.zip -d dataset/
    mv dataset/Univariate_ts dataset/UCR
fi


# Other timeseriesclassification.com datasets:

# Blink data
if [ ! -f dataset/Blink.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/Blink.zip" -O dataset/Blink.zip
    unzip dataset/Blink.zip -d dataset/Blink
fi

# MotionSenseHAR data
if [ ! -f dataset/MotionSenseHAR.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/MotionSenseHAR.zip" -O dataset/MotionSenseHAR.zip
    unzip dataset/MotionSenseHAR.zip -d dataset/MotionSenseHAR
fi

# EMOPain data
if [ ! -f dataset/EMOPain.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/EMOPain.zip" -O dataset/EMOPain.zip
    unzip dataset/EMOPain.zip -d dataset/EMOPain
fi


# SharePriceIncreasen data
if [ ! -f dataset/SharePriceIncrease.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/SharePriceIncrease.zip" -O dataset/SharePriceIncrease.zip
    unzip dataset/SharePriceIncrease.zip -d dataset/SharePriceIncrease
fi


# AbnormalHeartbeat data
if [ ! -f dataset/AbnormalHeartbeat.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/AbnormalHeartbeat.zip" -O dataset/AbnormalHeartbeat.zip
    unzip dataset/AbnormalHeartbeat.zip -d dataset/AbnormalHeartbeat
fi

# AsphaltObstacles data
if [ ! -f dataset/AsphaltObstacles.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/AsphaltObstacles.zip" -O dataset/AsphaltObstacles.zip
    unzip dataset/AsphaltObstacles.zip -d dataset/AsphaltObstacles
fi

# AsphaltObstaclesCoordinates data
if [ ! -f dataset/AsphaltObstaclesCoordinates.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/AsphaltObstaclesCoordinates.zip" -O dataset/AsphaltObstaclesCoordinates.zip
    unzip dataset/AsphaltObstaclesCoordinates.zip -d dataset/AsphaltObstaclesCoordinates
fi

# AsphaltPavementType data
if [ ! -f dataset/AsphaltPavementType.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/AsphaltPavementType.zip" -O dataset/AsphaltPavementType.zip
    unzip dataset/AsphaltPavementType.zip -d dataset/AsphaltPavementType
fi

# AsphaltPavementTypeCoordinates data
if [ ! -f dataset/AsphaltPavementTypeCoordinates.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/AsphaltPavementTypeCoordinates.zip" -O dataset/AsphaltPavementTypeCoordinates.zip
    unzip dataset/AsphaltPavementTypeCoordinates.zip -d dataset/AsphaltPavementTypeCoordinates
fi

# AsphaltRegularity data
if [ ! -f dataset/AsphaltRegularity.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/AsphaltRegularity.zip" -O dataset/AsphaltRegularity.zip
    unzip dataset/AsphaltRegularity.zip -d dataset/AsphaltRegularity
fi

# AsphaltRegularityCoordinates data
if [ ! -f dataset/AsphaltRegularityCoordinates.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/AsphaltRegularityCoordinates.zip" -O dataset/AsphaltRegularityCoordinates.zip
    unzip dataset/AsphaltRegularityCoordinates.zip -d dataset/AsphaltRegularityCoordinates
fi

# BinaryHeartbeat data
if [ ! -f dataset/BinaryHeartbeat.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/BinaryHeartbeat.zip" -O dataset/BinaryHeartbeat.zip
    unzip dataset/BinaryHeartbeat.zip -d dataset/BinaryHeartbeat
fi

# CatsDogs data
if [ ! -f dataset/CatsDogs.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/CatsDogs.zip" -O dataset/CatsDogs.zip
    unzip dataset/CatsDogs.zip -d dataset/CatsDogs
fi

# Colposcopy data
if [ ! -f dataset/Colposcopy.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/Colposcopy.zip" -O dataset/Colposcopy.zip
    unzip dataset/Colposcopy.zip -d dataset/Colposcopy
fi

# CounterMovementJump data
if [ ! -f dataset/CounterMovementJump.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/CounterMovementJump.zip" -O dataset/CounterMovementJump.zip
    unzip dataset/CounterMovementJump.zip -d dataset/CounterMovementJump
fi

# DucksAndGeese data
if [ ! -f dataset/DucksAndGeese.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/DucksAndGeese.zip" -O dataset/DucksAndGeese.zip
    unzip dataset/DucksAndGeese.zip -d dataset/DucksAndGeese
fi

# ElectricDeviceDetection data
if [ ! -f dataset/ElectricDeviceDetection.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/ElectricDeviceDetection.zip" -O dataset/ElectricDeviceDetection.zip
    unzip dataset/ElectricDeviceDetection.zip -d dataset/ElectricDeviceDetection
fi

# Epilepsy2 data
if [ ! -f dataset/Epilepsy2.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/Epilepsy2.zip" -O dataset/Epilepsy2.zip
    unzip dataset/Epilepsy2.zip -d dataset/Epilepsy2
fi

# EyesOpenShut data
if [ ! -f dataset/EyesOpenShut.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/EyesOpenShut.zip" -O dataset/EyesOpenShut.zip
    unzip dataset/EyesOpenShut.zip -d dataset/EyesOpenShut
fi

# FaultDetectionA data
if [ ! -f dataset/FaultDetectionA.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/FaultDetectionA.zip" -O dataset/FaultDetectionA.zip
    unzip dataset/FaultDetectionA.zip -d dataset/FaultDetectionA
fi

# FruitFlies data
if [ ! -f dataset/FruitFlies.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/FruitFlies.zip" -O dataset/FruitFlies.zip
    unzip dataset/FruitFlies.zip -d dataset/FruitFlies
fi

# InsectSound data
if [ ! -f dataset/InsectSound.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/InsectSound.zip" -O dataset/InsectSound.zip
    unzip dataset/InsectSound.zip -d dataset/InsectSound
fi

# KeplerLightCurves data
if [ ! -f dataset/KeplerLightCurves.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/KeplerLightCurves.zip" -O dataset/KeplerLightCurves.zip
    unzip dataset/KeplerLightCurves.zip -d dataset/KeplerLightCurves
fi

# MindReading data
if [ ! -f dataset/MindReading.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/MindReading.zip" -O dataset/MindReading.zip
    unzip dataset/MindReading.zip -d dataset/MindReading
fi

# MosquitoSound data
if [ ! -f dataset/MosquitoSound.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/MosquitoSound.zip" -O dataset/MosquitoSound.zip
    unzip dataset/MosquitoSound.zip -d dataset/MosquitoSound
fi

# NerveDamage data
if [ ! -f dataset/NerveDamage.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/NerveDamage.zip" -O dataset/NerveDamage.zip
    unzip dataset/NerveDamage.zip -d dataset/NerveDamage
fi

# RightWhaleCalls data
if [ ! -f dataset/RightWhaleCalls.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/RightWhaleCalls.zip" -O dataset/RightWhaleCalls.zip
    unzip dataset/RightWhaleCalls.zip -d dataset/RightWhaleCalls
fi

# Sleep data
if [ ! -f dataset/Sleep.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/Sleep.zip" -O dataset/Sleep.zip
    unzip dataset/Sleep.zip -d dataset/Sleep
fi

# Tiselac data
if [ ! -f dataset/Tiselac.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/Tiselac.zip" -O dataset/Tiselac.zip
    unzip dataset/Tiselac.zip -d dataset/Tiselac
fi

# UrbanSound data
if [ ! -f dataset/UrbanSound.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/UrbanSound.zip" -O dataset/UrbanSound.zip
    unzip dataset/UrbanSound.zip -d dataset/UrbanSound
fi

# WalkingSittingStanding data
if [ ! -f dataset/WalkingSittingStanding.zip ]; then
    wget "https://www.timeseriesclassification.com/aeon-toolkit/WalkingSittingStanding.zip" -O dataset/WalkingSittingStanding.zip
    unzip dataset/WalkingSittingStanding.zip -d dataset/WalkingSittingStanding
fi
