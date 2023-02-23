library(sharp)
library(argparse)

# Loading the arguments
parser <- ArgumentParser()
parser$add_argument("--path_data", help = "Path to data file")
parser$add_argument("--path_label", help = "Path to label file")
parser$add_argument("--indices", help = "Path to indices file", nargs='+')
parser$add_argument("--group", help = "Label group/column for directory")
parser$add_argument("--output", help = "Filename of output with full directory path")

args <- parser$parse_args()

path_to_dat <- args$path_data
path_to_label <- args$path_label
path_to_indices <- args$indices
group <- args$group
path_to_output <- args$output

###
# 01 Stability analysis LASSO ----------------------------------------------------------
###
output_dir <- dirname(path_to_output)
#dir.create(file.path(output_dir), recursive = TRUE)

mydat <- read.csv(path_to_dat, row.names = "X")
mylabels <- read.csv(path_to_label, row.names = "X")

for (i in length(path_to_indices)) {
  feature_to_remove <- read.csv(path_to_indices[i])
  mydat <- mydat[, !(colnames(mydat) %in% feature_to_remove$feature)]
}

print(dim(mydat))

if (ncol(mylabels) == 1) {
  outcome <- mylabels[row.names(mydat),]
} else {
  outcome <- eval(parse(text=paste0("mylabels[row.names(mydat),]$", group)))
}

head(outcome)

###
# Variable selection
###
stability_analysis <- function(xdata, ydata, family="gaussian", penalty=NULL, suffix="", dir = "") {
  
  if(is.null(penalty)) {
    stab=VariableSelection(xdata, ydata, family=family,pi_list = seq(0.6, 0.99, by = 0.01),
                           Lambda = LambdaSequence(lmax = 0.1, lmin = 1e-10, cardinal = 100),
                           tau = 0.8)
  } else {
    stab=VariableSelection(xdata, ydata, family=family,pi_list = seq(0.6, 0.99, by = 0.01),
                           penalty.factor = penalty,
                           Lambda = LambdaSequence(lmax = 0.1, lmin = 1e-10, cardinal = 100),
                           tau = 0.8)
  }
  
  calprop_out <- paste0(dir, "/calibration_plot_", suffix, ".png")
  cat("Printing figure to", calprop_out, "\n")
  png(calprop_out, width = 1200, height= 800, res = 150)
  grid::grid.newpage()
  grid::grid.draw(CalibrationPlot(stab))
  dev.off()
  
  selprop=SelectionProportions(stab)
  selprop_ranked=sort(selprop, decreasing=TRUE)
  
  selprop_out <-  paste0(dir, "/selection_proportion_", suffix, ".png")
  cat("Printing figure to", selprop_out, "\n")
  png(selprop_out, width = 1200, height= 800, res = 150)
  par(mar=c(10,6,2,2))
  plot(selprop_ranked, type="h", lwd=3, las=1, cex.lab=1.3, bty="n", ylim=c(0,1),
       col=ifelse(selprop_ranked>=Argmax(stab)[2],yes="red",no="grey"),
       xaxt="n", xlab="", ylab="Selection proportions")
  abline(h=Argmax(stab)[2], lty=2, col="darkred")
  axis(side=1, at=1:length(selprop_ranked), labels=names(selprop_ranked), las=2)
  dev.off()

  
    selected <- SelectedVariables(stab)
    p <- sum(selected)
    features <- names(selprop)[sort.list(selprop, decreasing = TRUE)][1:p]
    features <- gsub("[.]", " ", features)
    features <- data.frame(feature = features)

    write.csv(features, path_to_output, row.names = F)
  
  return(stab)
}

stab <- stability_analysis(mydat, outcome, family="binomial", penalty = NULL, dir = output_dir)
saveRDS(stab, paste0(output_dir,"/stab.rds"))


