library(sharp)
library(argparse)

# Function to determine the mode of the target variable
get_mode <- function(y_train) {
  if (all(sapply(y_train, is.character))) {
    unique_values <- unique(y_train)
    if (length(unique_values) == 2) {
      return("Classification")
    } else {
      return("Classification")
    }
  } else if (all(sapply(y_train, is.numeric))) {
    if (all(sapply(y_train, function(item) item == 0 || item == 1))) {
      return("Classification")
    } else {
      return("Regression")
    }
  }
}

# Loading the arguments
parser <- ArgumentParser()
parser$add_argument("--path_data", help = "Path to data file")
parser$add_argument("--file_label", help = "Path to label file")
parser$add_argument("--indices", help = "Path to indices file", nargs='+')
parser$add_argument("--group", help = "Label group/column for directory")
parser$add_argument("--output", help = "Filename of output with full directory path")

args <- parser$parse_args()

path_to_dat <- args$path_data
path_to_label <- args$file_label
path_to_indices <- args$indices
group <- args$group
path_to_output <- args$output

# Stability analysis LASSO
output_dir <- dirname(path_to_output)
mydat <- read.csv(path_to_dat, row.names = 1, check.names=FALSE)
mylabels <- read.csv(path_to_label, row.names = 1, check.names=FALSE)

# Remove features based on indices file
for (i in length(path_to_indices)) {
  feature_to_remove <- read.csv(path_to_indices[i])
  mydat <- mydat[, !(colnames(mydat) %in% feature_to_remove$feature)]
}

df <- merge(mylabels, mydat, by = "row.names")
rownames(df) <- df$Row.names
df$Row.names <- NULL
df <- df[which(!is.na(df[,1])), ]
mydat <- df[, 2:ncol(df)]

print(dim(mydat))
head(mydat)

# Get the outcome variable based on the label file and group
if (ncol(mylabels) == 1) {
  outcome <- mylabels[row.names(mydat),]
} else {
  outcome <- eval(parse(text=paste0("mylabels[row.names(mydat),]$", group)))
}

head(outcome)

# Determine the mode (classification or regression)
mode = get_mode(outcome)
print(mode)

if (mode == "Classification") {
  family = "binomial"
} else if (mode == "Regression") {
  family = "gaussian"
}
print(family)

# Variable selection using stability analysis
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
  
  # Create and save calibration plot
  calprop_out <- paste0(dir, "/calibration_plot_", suffix, ".png")
  cat("Printing figure to", calprop_out, "\n")
  print(head(stab$S_2d))
  print(head(stab$selprop))
  png(calprop_out, width = 1200, height= 800, res = 150)
  grid::grid.newpage()
  grid::grid.draw(CalibrationPlot(stab))
  dev.off()
  
  # Create and save selection proportions plot
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
  
  # Get the selected variables and save them to the output file
  selected <- SelectedVariables(stab)
  p <- sum(selected)
  features <- names(selprop)[sort.list(selprop, decreasing = TRUE)][1:p]
  features <- data.frame(feature = features)
  
  write.csv(features, path_to_output, row.names = F)
  
  return(stab)
}

stab <- stability_analysis(mydat, outcome, family=family, penalty = NULL, dir = output_dir)
saveRDS(stab, paste0(output_dir,"/stab.rds"))
