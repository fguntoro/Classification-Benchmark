library(sharp)
library(argparse)

# Loading the arguments
args <- commandArgs(trailingOnly = TRUE)
path_to_dat <- as.character(args[1])
path_to_label <- as.character(args[2])
path_to_indices <- as.character(args[3])
group <- as.character(args[4])
path_to_output <- as.character(args[5])

###
# 01 Stability analysis LASSO ----------------------------------------------------------
###
output_dir <- dirname(path_to_output)
dir.create(file.path(output_dir), recursive = TRUE)

mydat <- read.csv(path_to_dat, row.names = "X")
mylabels <- read.csv(path_to_label, row.names = "X")
myindices <- read.csv(path_to_indices)
mydat <- mydat[myindices$index,]
print(dim(mydat))

if (ncol(mylabels) == 1) {
  outcome <- mylabels[row.names(mydat),]
} else {
  outcome <- eval(parse(text=paste0("mylabels[row.names(mydat),]$", group)))
}

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
    all_variables <- data.frame(feature=(names(selprop)[sort.list(selprop, decreasing = TRUE)])[1:p])
    #data_filtered <- xdata[,all_variables]

    write.csv(all_variables, paste0(output_dir,"/stability.csv"), row.names=F)
  
  return(stab)
}

stab <- stability_analysis(mydat, outcome, family="binomial", penalty = NULL, dir = output_dir)
saveRDS(stab, paste0(output_dir,"/stab.rds"))


