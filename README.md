# IGNITE
IGNITE: Individualized GeNeration of Imputations in Time-series Electronic health records

Table of contents
=================

<!--ts-->
   * [About the repository](#About-the-repository)
      * [Background](#Background)
      * [Code](#Code)
   * [Citation](#Citation)
   
<!--te-->

About the repository
============
## Background
 
Time-series Electronic Health Record (EHR) data are very sparse and highly missing. If missingness is not properly represented, the data utility, subsequent analysis and performance of downstream applications might get significantly impacted. These challenges become of higher complexity when the time-series data are multivariate and contain mixed-type courses of treatment and continuous-valued physiological data.
In this work, we propose a novel hybrid deep-generative model for imputing personalized missing data by incorporating individualized missingness patterns and discrete treatments over time. Our IGNITE model (Individualized GeNeration of Imputations in Time-series Electronic health records) utilizes a dual-variational autoencoder conditioned on the personalised treatments, augmented with a novel individualized missingness mask that accounts for individual-level differences in missingness frequencies across the multivariate time series. They are then complemented with a discriminator to improve the quality of the imputations. We test our model on a publicly-available intensive care unit dataset and show that IGNITE outperforms state-of-the-art imputation methods in downstream tasks.

## Code 



## Citation


If you found this code useful, please cite: ...... TO ADD
