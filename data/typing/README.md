##Entity Typing Dataset


Etrain, Edev and Etest are the entity-types datasets. 
Format of each file is:

entity_mid <TAB> notable_type <TAB> types(SPACE-seperated) <TAB> freq <TAB> #### <TAB> name1 <TAB> freqOfName1 name2 <TAB> freqOfName2 ... 

For example:

/m/01mngw -location-city  -location-city  -location 327 ####  Hudson  352 HUDSON  5


In the experiments, each entity is annotated with all types. E.g., entity (/m/01mngw) has (-location-city, -location) as gold types. 

File "types" contains the set of types in the dataset. 

