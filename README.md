# A Transformer Approach to Air Pollution Forecast for the Arctic and Northern Europe
Cuzzucoli A., Crotti I., Dobricic S., Pasini A.

This repository shows the supporting code for the manuscript available at:

    https://doi.org/10.1038/s44407-026-00071-8

# Summary
We developed a Deep Learning model to predict 48h local concentrations of $\textrm{PM}_{10}$ at monitoring stations sites across Northern Europe. 

We considered historical time-series data from the European Enviromental Agency and hourly forecasts of $\textrm{PM}_{10}$ and meteorological variables from the Copernicus Atmosphere Monitoring Service.

# Data
Data was collected from 152 monitoring stations across Northern Europe.
![monitoring stations](/img/NE-stations.png)

Input variables include:

- hourly $\textrm{PM}_{10}$ concentration from monitoring stations
- 48h $\textrm{PM}_{10}$ concentration forecasts at hourly resolution from deterministic models released from the Copernicus Atmosphere Monitoring Service (CHIMERE, DEHM,EMEP, EURAD-IM, GEMA-Q, MATCH, MOCAGE, SILAM)
- 48h meteorological variables forecast from the Copernicus Atmosphere Monitoring Service (Temperature at 2 m, U and V components of the wind, Boundary Layer Height, Mean Sea Level Pressure, Total Precipitation)

Links for data sources can be found at [references](/demo_dataset/references_demo_dataset.txt).


All information for the stations can be found in [metadata](/stations_metadata.csv).

A sample dateset can be found at [demo](/demo_dataset/SPO-SE0003A_00005_100_data.csv).

# Model Architecture
![model](/img/acf-architecture.png)
The Adapted Crossformer represents an enhancement of the original [Crossformer](https://openreview.net/forum?id=vSVLM2j9eie) Architecture, which considers a Two-Stage Attention Layer (cross-time and cross-dimension). In addition, the Adapted Crossformer embeds geographical coordinates of the monitoring stations together with input data taken from the specified station.

# Repository Structure
```bash
├── adaptedCF-env.yml
├── data_manager
│   ├── data_framer.py
│   └── data_loader.py
├── demo_dataset
│   ├── references_demo_dataset.txt
│   └── SPO-SE0003A_00005_100_data.csv
├── img
│   ├── acf-architecture.png
│   └── NE-stations.png
├── LICENSE
├── model
│   ├── attention_layer.py
│   ├── crossformer.py
│   ├── decoder_layer.py
│   ├── embedding_layer.py
│   └── encoder_layer.py
├── parameters.py
├── pretrained_model
│   └── CF-GDSW_stall_shift48_proj-True_dim31_inlen96_outlen48_seglen24_win2_f6_dmodel256_dff512_nheads4_elayers3_drop0.2_ep20_p3_0.5-0.25-0.25split_segm-True_huber
│       └── checkpoint.pth
├── process.py
├── README.md
├── requirements.txt
├── stations_metadata.csv
├── test.py
├── training.py
└── utils
    ├── metrics.py
    ├── spatial.py
    └── tools.py
```

# Usage

To train Adapted Crossformer on sample station with standard parameters:

    python process.py

Configurations and Gridsearch for tuning can be set in [parameters](/parameters.py).

To evaluate model performance using pretrained weights from 1/1/2024 to 4/12/2024:

    python process.py --eval

Start and end date can be selected by setting process arguments <code>--start_date \[dict]</code> and <code>--end_date \[dict]</code>.

Pretrained weights of Adapted Crossformer with optimised parameters can be found at [weights](/pretrained_model/CF-GDSW_stall_shift48_proj-True_dim31_inlen96_outlen48_seglen24_win2_f6_dmodel256_dff512_nheads4_elayers3_drop0.2_ep20_p3_0.5-0.25-0.25split_segm-True_huber/checkpoint.pth).

# How to cite

Manuscript

    Cuzzucoli, A., Crotti, I., Dobricic, S. et al. A transformer approach to forecasting PM10 concentration in the Arctic and Northern Europe. npj Clean Air 2, 31 (2026). https://doi.org/10.1038/s44407-026-00071-8

Bibtex

    @ARTICLE{Cuzzucoli2026-yn,
      title     = "A transformer approach to forecasting {PM10} concentration in
                   the Arctic and Northern Europe",
      author    = "Cuzzucoli, Alice and Crotti, Ilaria and Dobricic, Srdjan and
                   Pasini, Antonello",
      abstract  = "Abstract Air pollution is reckoned as one of the main critical
                   factors in risks related to human health as well as to the
                   environment, especially in key areas like the Arctic, where
                   global warming is affecting the ecosystem at an alarming rate.
                   Major pollutants, such as Particulate Matter (PM 10 ), are thus
                   crucial to monitor for efficient planning ahead of extreme
                   events: to that extent, a reliable short to medium-term forecast
                   is vital for prompt action. We propose a Deep Learning approach
                   to air pollution forecasting aimed at predicting 48-h
                   concentrations of PM 10 . We analyse the behaviour of said
                   pollutant together with auxiliary factors, such as
                   meteorological variables and outputs from state-of-the-art
                   numerical models, through several forefront
                   time-series-specialised architectures, i.e. Transformers. We
                   develop an enhanced version of the best-performing Transformer
                   architecture to define a suitable model to forecast PM 10
                   concentrations from a wide variety of stations across Northern
                   Europe. Our proposed methodology manages to outperform pivotal
                   numerical models and associated postprocessing, offering a
                   valuable alternative for local air pollution forecasting.",
      journal   = "NPJ Clean Air",
      publisher = "Springer Science and Business Media LLC",
      volume    =  2,
      number    =  1,
      month     =  may,
      year      =  2026,
      copyright = "https://creativecommons.org/licenses/by/4.0",
      language  = "en"
    }

Code

    https://doi.org/10.5281/zenodo.19354381
