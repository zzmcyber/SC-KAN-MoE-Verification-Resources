# Data sources

The auxiliary subset retains actual project inputs with anonymous identifiers and privacy-shifted timestamps.

| Provider / product family | Released inputs and numerical convention | Source |
| --- | --- | --- |
| Sentinel-1 / Copernicus | VV/VH in dB; single-pixel linear power; 3 × 3 means in linear power; angles in degrees; SAR quality flags | [Sentinel-1](https://sentiwiki.copernicus.eu/web/s1-mission) |
| ERA5-Land / ECMWF | Temperatures in K; accumulated radiation in J/m2; evaporation and snow quantities in m; wind speed in m/s; freeze-thaw flag | [ERA5-Land](https://www.ecmwf.int/en/era5-land) |
| GPM IMERG / NASA | Precipitation accumulations in mm over 1 h, 3 h, 6 h, 24 h, 3 d and 7 d; elapsed time since rain in hours | [IMERG](https://gpm.nasa.gov/data/imerg) |
| MODIS / NASA | NDVI, EVI and FPAR dimensionless; LAI in m2/m2; temporal-matching intervals in days | [MODIS](https://modis.gsfc.nasa.gov/data/) |
| Copernicus DEM | Terrain elevation in m | [Copernicus DEM](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM) |

SAR fields are project-processed auxiliary inputs. The neighborhood fields are linear-domain means. Radiation retains accumulated-energy units, and evaporation retains the provider's sign convention. Scientific values are copied without numerical adjustment.

## Acknowledgments

Contains modified Copernicus Sentinel data. Contains modified Copernicus Climate Change Service information. The European Commission and ECMWF are not responsible for use of the Copernicus information in these resources.

GPM IMERG and MODIS inputs are provided by NASA and its data centers.

Copernicus DEM attribution: © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.
