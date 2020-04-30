library(tidyverse)
library(dyngen)

set.seed(10)
model <- initialise_model(
  num_tfs = 12,
  num_targets = 30,
  num_hks = 15,
  backbone = backbone_linear(),
  verbose = TRUE,
  download_cache_dir = "~/.cache/dyngen",
  num_cores = 8
)

dataset_model <- generate_dataset(model)

dataset <- readRDS('/data/causal/datadataset.rds')
#dataset <- dataset_model$dataset
model <- readRDS('/data/causal/datamodel.rds')
#model <- dataset_model$model
plot_feature_network(model, show_hks = TRUE)
#counts
counts <- t(dataset$expression)
#time
time <- dataset$progressions$percentage
milestones <- dataset$progressions$from
ii <- milestones == "sA"
milestones <- milestones[ii]
time <- time[ii]
counts <- counts[,ii]
time <- time / length(unique(milestones))
k = 0
for(i in sort(unique(milestones))){
  time[milestones == i] <- time[milestones == i] + 1/length(unique(milestones)) * k
  k = k + 1
}
time <- time / max(time)
ii <- order(time)
counts <- counts[,ii]
time <- time[ii] * 100

library(reticulate)
np <- import("numpy")
np$savez("/data/causal/data/simulated_dataset.npz",mat=counts,ptime=time,gene_names=rownames(counts))

library(R.matlab)
filename <- "/data/causal/data/simulated_dataset.mat"
writeMat(filename,X=counts,ptime=time)

    # library(monocle)
# fdata <- data.frame(gene_short_name=colnames(dataset$counts))
# fd <- new("AnnotatedDataFrame", data = fdata)
# rownames(fd) <- colnames(dataset$counts)
# monocle_data <- newCellDataSet(t(as.matrix(dataset$counts)),featureData = fd)
# 
# diff_test_res <- differentialGeneTest(monocle_data)
# ordering_genes <- row.names (subset(diff_test_res, qval < 0.01))