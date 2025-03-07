library(UniDOE)
library(parallel)

# Set working directory
output_dir <- "C:/Users/zzh/Desktop/data1"
if (!dir.exists(output_dir)) {  
  dir.create(output_dir)
}

# Define the list of amino acids
amino_acids <- c("A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", 
                 "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V")

# Loop to generate tetrapeptide sequences and save to CSV file
for (n in seq(1000, 16000, by=500)) {  
  s <- 4  
  q <- 20  
  crit <- "MD2"    
  
  # Record the start time
  start_time <- Sys.time()
  cat("Starting generation for n =", n, "at", format(start_time, "%Y-%m-%d %H:%M:%S"), "\n")    
  
  # Create and use a parallel cluster
  c1 <- makeCluster(detectCores() - 1)
  clusterExport(c1, c("n", "s", "q", "crit", "GenUD"))
  res <- parLapply(c1, 1, function(x) GenUD(n, s, q, crit=crit, maxiter=100))[[1]]
  stopCluster(c1)    
  
  # Record the end time and calculate elapsed time
  end_time <- Sys.time()
  cat("Generation completed for n =", n, "at", format(end_time, "%Y-%m-%d %H:%M:%S"), "\n")
  cat("Time taken:", end_time - start_time, "\n")    
  
  # Convert generated samples into tetrapeptide sequences
  generated_tetrapeptides <- apply(res$final_design, 1, function(x) paste(amino_acids[x], collapse=""))    
  
  # Save to CSV file
  output_file <- file.path(output_dir, paste0("tetrapeptides_", n, ".csv"))
  write.csv(generated_tetrapeptides, file=output_file, row.names=FALSE)    
  
  # Calculate the frequency of each amino acid at each position
  position_frequencies <- matrix(0, nrow=length(amino_acids), ncol=s)
  rownames(position_frequencies) <- amino_acids
  colnames(position_frequencies) <- paste0("Position_", 1:s)    
  
  for (i in 1:s) {    
    position_frequencies[, i] <- table(factor(sapply(generated_tetrapeptides, function(seq) strsplit(seq, "")[[1]][i]), levels=amino_acids))  
  }    
  
  position_frequencies <- position_frequencies / n    
  
  # Plot the frequency of each amino acid at each position and save the plot
  matplot(t(position_frequencies), type="l", lty=1, col=1:20,          
          main=paste("Frequency of Amino Acids at Each Position (n=", n, ")", sep=""),          
          xlab="Position", ylab="Frequency", xaxt='n')  
  axis(1, at=1:s, labels=colnames(position_frequencies))  
  legend("topright", legend=rownames(position_frequencies), col=1:20, lty=1, cex=0.8)    
  
  # Save the plot file
  plot_file <- file.path(output_dir, paste0("frequency_plot_", n, ".png"))
  dev.copy(png, plot_file)
  dev.off()    
  
  cat("Results saved for n =", n, "\n\n")
}
