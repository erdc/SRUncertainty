## Super-resolution and Uncertainty Estimation from Sparse Sensors of Dynamical Physical Systems

This is supporting code for the article
```
Collins, A.M.; Rivera-Casillas, P.; Dutta, S.; Cecil, O.M.; Trautz, A.C.; Farthing, M.W.
Super-resolution and uncertainty estimation from sparse sensors of dynamical physical systems.
Frontiers in Water, 2023, (Under Review)
```

Email: adam.m.collins@erdc.dren.mil for any questions/feedback.

Super-resolution framework
:-----:
<p align="center">
    <img align = 'center' height="500" src="figures/unet.png?raw=true">
</p>


## Getting Started

* Download the relevant datasets from the following links and save them in respective subdirectories under the `data` directory.
  - [Link](https://www.kaggle.com/datasets/ryleymcconkey/ml-turbulence-dataset) for the mean flow dataset. Individual cases to be saved as `data/kepsilon`, `data/komega` etc.
  - [Link](https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.html) for the NOAA Optimal SST dataset. To be saved in `data/noaa/`.
  - [Link](https://doi.org/10.6084/m9.figshare.c.5142185.v1) for the global soil moisture dataset. To be saved in `data/sm/`.
* The first run for each example generates relevant tessellated fields and saves them in subdirectories under `data/npy/`.

### Dependencies

* Python 3.x
* Tensorflow TF 2.x. Install either the CPU or the GPU version depending on available resources.
* A list of all the dependencies are provided in the [requirements](requirements.txt) file.

### Executing program

* Training and evaluation for the mean flow dataset can be performed using  `examples/Mean_Flow/mf.py`. The script is set up to parse various arguments during the execution call which typically looks like
`python mf.py 'unet-kepsilon-PHLL_case_0p5 PHLL_case_0p8 PHLL_case_1p0 PHLL_case_1p2 BUMP_h20 BUMP_h26 BUMP_h31 BUMP_h38 CNDV_12600-Ux Uy-PHLL_case_1p5 BUMP_h42 CNDV_20580-Ux Uy-train'`
* Training and evaluation for the NOAA SST dataset can be performed using `examples/Sea_Surface_Temperature/sst.py`.
* Training and evaluation for the soil moisture dataset can be performed using `examples/Soil_Moisture/soilmoisture.py`.
