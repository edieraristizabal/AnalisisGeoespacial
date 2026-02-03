#Linear regression

<https://andrewmaclachlan.github.io/CASA0005repo/explaining-spatial-patterns.html>

#download a zip file containing some boundaries we want to use
download.file("https://data.london.gov.uk/download/statistical-gis-boundary-files-london/9ba8c833-6370-4b11-abdc-314aa020d5e0/statistical-gis-boundaries-london.zip", 
              destfile="prac7_data/statistical-gis-boundaries-london.zip")

library(fs)
listfiles<-dir_info(here::here("prac7_data")) %>%
  dplyr::filter(str_detect(path, ".zip")) %>%
  dplyr::select(path)%>%
  pull()%>%
  #print out the .gz file
  print()%>%
  as.character()%>%
  utils::unzip(exdir=here::here("prac7_data"))

#look what is inside the zip

Londonwards<-fs::dir_info(here::here("prac7_data","statistical-gis-boundaries-london","ESRI"))%>%
  #$ means exact match
  dplyr::filter(str_detect(path, 
                           "London_Ward_CityMerged.shp$"))%>%
  dplyr::select(path)%>%
  dplyr::pull()%>%
  #read in the file in
  sf::st_read()

#check the data
qtm(Londonwards)

#read in some attribute data
LondonWardProfiles <- read_csv("https://data.london.gov.uk/download/ward-profiles-and-atlas/772d2d64-e8c6-46cb-86f9-e52b4c7851bc/ward-profiles-excel-version.csv", 
                               col_names = TRUE, 
                               locale = locale(encoding = 'Latin1'))

#check all of the columns have been read in correctly
Datatypelist <- LondonWardProfiles %>% 
  summarise_all(class) %>%
  pivot_longer(everything(), 
               names_to="All_variables", 
               values_to="Variable_class")

Datatypelist

#We can use readr to deal with the issues in this dataset - which are to do with text values being stored in columns containing numeric values

#read in some data - couple of things here. Read in specifying a load of likely 'n/a' values, also specify Latin1 as encoding as there is a pound sign (£) in one of the column headers - just to make things fun!

LondonWardProfiles <- read_csv("https://data.london.gov.uk/download/ward-profiles-and-atlas/772d2d64-e8c6-46cb-86f9-e52b4c7851bc/ward-profiles-excel-version.csv", 
                               na = c("", "NA", "n/a"), 
                               locale = locale(encoding = 'Latin1'), 
                               col_names = TRUE)

#check all of the columns have been read in correctly
Datatypelist <- LondonWardProfiles %>% 
  summarise_all(class) %>%
  pivot_longer(everything(), 
               names_to="All_variables", 
               values_to="Variable_class")

Datatypelist

#merge boundaries and data
LonWardProfiles <- Londonwards%>%
  left_join(.,
            LondonWardProfiles, 
            by = c("GSS_CODE" = "New code"))

#let's map our dependent variable to see if the join has worked:
tmap_mode("plot")
qtm(LonWardProfiles, 
    fill = "Average GCSE capped point scores - 2014", 
    borders = NULL,  
    fill.palette = "Blues")

#might be a good idea to see where the secondary schools are in London too
london_schools <- read_csv("https://data.london.gov.uk/download/london-schools-atlas/57046151-39a0-45d9-8dc0-27ea7fd02de8/all_schools_xy_2016.csv")

#from the coordinate values stored in the x and y columns, which look like they are latitude and longitude values, create a new points dataset
lon_schools_sf <- st_as_sf(london_schools, 
                           coords = c("x","y"), 
                           crs = 4326)

lond_sec_schools_sf <- lon_schools_sf %>%
  filter(PHASE=="Secondary")

tmap_mode("plot")
qtm(lond_sec_schools_sf)

#plot with a regression line - note, I've added some jitter here as the x-scale is rounded
q + stat_smooth(method="lm", se=FALSE, size=1) + 
  geom_jitter()

#run the linear regression model and store its outputs in an object called model1
Regressiondata<- LonWardProfiles%>%
  clean_names()%>%
  dplyr::select(average_gcse_capped_point_scores_2014, 
                unauthorised_absence_in_all_schools_percent_2013)

#now model
model1 <- Regressiondata %>%
  lm(average_gcse_capped_point_scores_2014 ~
       unauthorised_absence_in_all_schools_percent_2013,
     data=.)

#show the summary of those outputs
summary(model1)

library(broom)
tidy(model1)

glance(model1)

library(tidypredict)
Regressiondata %>%
  tidypredict_to_column(model1)

#let's check the distribution of these variables first

ggplot(LonWardProfiles, aes(x=`Average GCSE capped point scores - 2014`)) + 
  geom_histogram(aes(y = ..density..),
                 binwidth = 5) + 
  geom_density(colour="red", size=1, adjust=1)

ggplot(LonWardProfiles, aes(x=`Unauthorised Absence in All Schools (%) - 2013`)) +
  geom_histogram(aes(y = ..density..),
                 binwidth = 0.1) + 
  geom_density(colour="red",
               size=1, 
               adjust=1)

library(ggplot2)

# from 21/10 there is an error on the website with 
# median_house_price_2014 being called median_house_price<c2>2014
# this was corrected around 23/11 but can be corrected with rename..

LonWardProfiles <- LonWardProfiles %>%
  #try removing this line to see if it works...
  dplyr::rename(median_house_price_2014 =`Median House Price (£) - 2014`)%>%
  janitor::clean_names()

ggplot(LonWardProfiles, aes(x=median_house_price_2014)) + 
  geom_histogram()

qplot(x = median_house_price_2014, 
          y = average_gcse_capped_point_scores_2014, 
          data=LonWardProfiles)

#save the residuals into your dataframe

model_data <- model1 %>%
  augment(., Regressiondata)

#plot residuals
model_data%>%
  dplyr::select(.resid)%>%
  pull()%>%
  qplot()+ 
  geom_histogram() 

#print some model diagnositcs. 
par(mfrow=c(2,2))    #plot to 2 by 2 array
plot(model2)

#calculate the centroids of all Wards in London
coordsW <- LonWardProfiles%>%
  st_centroid()%>%
  st_geometry()

plot(coordsW)

#Now we need to generate a spatial weights matrix 
#(remember from the lecture a couple of weeks ago). 
#We'll start with a simple binary matrix of queen's case neighbours

LWard_nb <- LonWardProfiles %>%
  poly2nb(., queen=T)

#or nearest neighbours
knn_wards <-coordsW %>%
  knearneigh(., k=4)

LWard_knn <- knn_wards %>%
  knn2nb()

#plot them
plot(LWard_nb, st_geometry(coordsW), col="red")

plot(LWard_knn, st_geometry(coordsW), col="blue")

#create a spatial weights matrix object from these weights

Lward.queens_weight <- LWard_nb %>%
  nb2listw(., style="W")

Lward.knn_4_weight <- LWard_knn %>%
  nb2listw(., style="W")

Queen <- LonWardProfiles %>%
  st_drop_geometry()%>%
  dplyr::select(model2resids)%>%
  pull()%>%
  moran.test(., Lward.queens_weight)%>%
  tidy()

Nearest_neighbour <- LonWardProfiles %>%
  st_drop_geometry()%>%
  dplyr::select(model2resids)%>%
  pull()%>%
  moran.test(., Lward.knn_4_weight)%>%
  tidy()

Queen

Nearest_neighbour