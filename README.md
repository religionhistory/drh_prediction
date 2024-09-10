# NB: 
This is a fork of "Imputation Comparison" (Rachel Spicer). 

# Imputation Comparison
Code used for the analysis described in **Predicting the Past: Imputation of Historical Data**. This analysis was [preregistered](https://osf.io/7q9jm). 

## Data 

Data were downloaded from the [Database of Religious History (DRH)](https://religiondatabase.org/landing/) on 26<sup>th</sup> October 2021 and consists of 511 entries with 95579 unique answers. 

## Data Analysis Software

All data preprocessing was performed using [R version 4.0.5](https://cran.r-project.org/index.html). The following packages were used:
  - [tidyverse version 1.3.1](https://cran.r-project.org/web/packages/tidyverse/index.html)
  - [data.table version 1.14.0](https://cran.r-project.org/web/packages/data.table/index.html)
  - [splitstackshape version 1.4.8](https://cran.r-project.org/web/packages/splitstackshape/index.html)
  - [testthat version 3.0.2](https://cran.r-project.org/web/packages/testthat/index.html)
  - [devtools version 2.4.1](https://cran.r-project.org/web/packages/devtools/index.html)

Imputation was performed using R version 4.0.5](https://cran.r-project.org/index.html) and Python 3.7.6. In python the module [datawig version 0.2.0]() was used for imputation, along with [pandas version 1.2.3](), [numpy version 1.20.1](), os and glob. The R packages used for imputation were as follows:
  - [mice version 3.13.0](https://cran.r-project.org/web/packages/mice/index.html)
  - [missForest version 1.4](https://cran.r-project.org/web/packages/missForest/index.html)
  - [VIM version 6.1.0](https://cran.r-project.org/web/packages/VIM/index.html)
  - [h2o version 3.32.1.3](https://cran.r-project.org/web/packages/h2o/index.html)
  - [missMDA version 1.18](https://cran.r-project.org/web/packages/missMDA/index.html)

Comparison of imputation methods was performed using [R version 4.0.5](https://cran.r-project.org/index.html). The following packages were used:
  - [tidyverse version 1.3.1](https://cran.r-project.org/web/packages/tidyverse/index.html)
  - [Metrics version 0.1.4](https://cran.r-project.org/web/packages/Metrics/index.html)

Figures were created using [R version 4.2.2](https://cran.r-project.org/index.html), with the following packages: 
  - [ggplot2 version 3.4.1](https://cran.r-project.org/web/packages/ggplot2/index.html)
  - [ggpubr version 0.6.0](https://cran.r-project.org/web/packages/ggpubr/index.html)
  - [ggrepel version 0.9.3](https://cran.r-project.org/web/packages/ggrepel/index.html)
