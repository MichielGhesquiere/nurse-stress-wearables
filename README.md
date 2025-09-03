# Nurse Stress Prediction Wearable Sensors

*A multimodal sensor dataset for continuous stress detection of nurses in a hospital*

**Kaggle Dataset:**  
https://www.kaggle.com/datasets/priyankraval/nurse-stress-prediction-wearable-sensors/data

---

## Table of Contents
1. Introduction
2. Project Overview
3. Dataset Description
    - Data Collection Context
    - Data Captured
    - Study Procedure
    - Data Availability
4. Merge CSV File Information

---

## Introduction

The growing accessibility of wearable tech has opened doors to continuously monitor various physiological factors. Detecting stress early has become pivotal, aiding individuals in proactively managing their health against the detrimental effects of prolonged stress exposure. This dataset was cultivated within the natural environment of a hospital during the COVID-19 outbreak, encompassing the biometric data of nurses. Our dataset not only encompasses physiological data but also contextual information surrounding stress events.

---

## Project Overview

This project delves into leveraging wearable device-derived physiological signals to gauge stress levels among nurses operating within a hospital environment.

**Goals:**
- Evaluate various machine learning models to forecast stress levels based on recorded physiological signals.
- Investigate the most pertinent physiological indicators for stress detection.
- Offer insights to enhance the accuracy and dependability of stress detection via wearable tech.

---

## Dataset Description

### Data Collection Context
- **Period:** Data gathered over one week from 15 female nurses aged 30 to 55 years, during regular shifts at a hospital.
- **Collection Phases:**  
  - Phase-I: April 15, 2020, to August 6, 2020  
  - Phase-II: October 8, 2020, to December 11, 2020
- **Exclusion Criteria:** Pregnancy, heavy smoking, mental disorders, chronic or cardiovascular diseases.

### Data Captured
- **Physiological Variables Monitored:** Electrodermal activity, Heart Rate, and skin temperature.
- **Survey Responses:** Periodic smartphone-administered surveys capturing contributing factors to detected stress events.
- **Measurement Technologies:** Empatica E4 for Galvanic Skin Response and Blood Volume Pulse (BVP) readings.

### Study Procedure
- **Approval:** University's Institutional Review Board approved the study protocol (FA19–50 INFOR).
- **Consent and Enrollment:** Nurse subjects were enrolled after expressing interest and obtaining hospital compliance.
- **Study Design:** Conducted in three phases, each including 7 nurses. No incentives were provided, and anonymization of data was ensured.

### Data Availability
- **Public Release:** Database containing signals, stress events, and survey responses is publicly available on Dryad.
- **Anonymization:** Unique identifiers assigned to subjects to maintain anonymity.

---

## Merge CSV File Information

This dataset comprises approximately 11.5 million entries across nine columns:

- **X, Y, Z:** Orientation data (256 unique entries each)
- **EDA, HR, TEMP:** Physiological measurements (EDA: 274,452 unique, HR: 6,268 unique, TEMP: 599 unique)
- **id:** 18 categorical identifiers
- **datetime:** Extensive date and time entries (10.6 million unique)
- **label:** Categorical states or classes (three unique entries)

The dataset offers a wide array of continuous physiological measurements alongside orientation data, facilitating stress detection, health monitoring, and related research endeavours.