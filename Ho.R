library(raster)
library(rgdal)

args <- commandArgs(trailingOnly=TRUE)

RnTiff <- raster(args[1])
GTiff <- raster(args[2])
HoTiff <- GTiff
HoTiff[] <- RnTiff[] - GTiff[]

print(HoTiff)

writeRaster(HoTiff, filename="/home/ubuntu/Workspace/fast-sebal/HO.tif", format="GTiff", overwrite=TRUE)
