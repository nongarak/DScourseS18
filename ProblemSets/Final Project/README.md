To execute this project, you need to do a few things first. 

None of the data that are used are easily accessible/scrapable/automatable. 

1) To get Oklahoma Mesonet data, use the Daily Data Retrieval tool on the Oklahoma Mesonet website. 
https://www.mesonet.org/index.php/weather/daily_data_retrieval
Enter Jan 1, 1994 to Jan 1, 2018, select all stations and ATOT variable, and submit the request. 
Then, wait until you recieve an email with a link to download the requested data. 

2) To get DEM data, you can download it from the National Elevation Dataset on the USGS website, although 
I am not entirely sure how to do that. I recieved my data from my GIS instructor. 

3) To get land use data, use the USGS GAP Land Cover database downloader
https://gapanalysis.usgs.gov/gaplandcover/data/download/
Download data for Oklahoma, preprocess the data in ArcGIS, export the data as a .tif that can be used in R. 

4) To get median home value data, use the US Census American FactFinder interface to download 
median home data by census tract for the state of Oklahoma in .shp format. Open this in ArcGIS and export as a .tif file. 
https://factfinder.census.gov/faces/nav/jsf/pages/index.xhtml


Oklahoma Mesonet data are all ready to go, but the rest of the data need to be preprocessed in 
ArcGIS or a similar GIS program that can convert data files to .tif format for use in R. 
While R may be able to handle the different types of data using specific packages, for our 
analysis it was easier to convert them all to the same format. 

Once the data sources are prepared, run the finalproject.rmd file and the entire analysis should compile. 

Thanks for a great semester!
Alex




