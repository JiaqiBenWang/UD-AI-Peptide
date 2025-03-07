library(protr)
library(readxl)
library(writexl)
library(Peptides)
library(dplyr)

file_path <- "C:/Users/zzh/Desktop/16w/Supplementary Data 1.XLSX"
data <- read_excel(file_path)

head(data)

# Part 1: Amino acid composition

## Step 1: (Amino acid composition, AAC)
aac_matrix <- t(sapply(data$pep0, extractAAC))
data <- cbind(data, aac_matrix)

## Step 2: (Dipeptide composition, DC)
dc_matrix <- t(sapply(data$pep0, extractDC))
data <- cbind(data, dc_matrix)

head(data)


# Part 2: Autocorrelation

properties <- c("CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102", 
                "CHOC760101", "BIGC670101", "CHAM810101")

# Step 1: Extract Moreau-Broto autocorrelation features, set lag parameter
moreau_broto_matrix <- t(sapply(data$pep0, function(x) extractMoreauBroto(x, props = properties, nlag = 3L, customprops = NULL)))
colnames(moreau_broto_matrix) <- paste0("MoreauBroto_", colnames(moreau_broto_matrix))
data <- cbind(data, moreau_broto_matrix)

# Step 2: Extract Moran autocorrelation features, set lag parameter
moran_matrix <- t(sapply(data$pep0, function(x) extractMoran(x, props = properties, nlag = 3L, customprops = NULL)))
colnames(moran_matrix) <- paste0("Moran_", colnames(moran_matrix))
data <- cbind(data, moran_matrix)

# Step 3: Extract Geary autocorrelation features, set lag parameter
geary_matrix <- t(sapply(data$pep0, function(x) extractGeary(x, props = properties, nlag = 3L, customprops = NULL)))
colnames(geary_matrix) <- paste0("Geary_", colnames(geary_matrix))
data <- cbind(data, geary_matrix)

head(data)


# Part 3: CTD descriptors
# Composition
ctdc_matrix <- t(sapply(data$pep0, extractCTDC))
data <- cbind(data, ctdc_matrix)

# Transition
ctdt_matrix <- t(sapply(data$pep0, extractCTDT))
data <- cbind(data, ctdt_matrix)

# Distribution
ctdd_matrix <- t(sapply(data$pep0, extractCTDD))
data <- cbind(data, ctdd_matrix)

colnames(data)

# PArt 4: Quasi-sequence-order descriptors
# Step 1: (Sequence-Order-Coupling Numbers)
socn_matrix <- t(sapply(data$pep0, extractSOCN, nlag = 3))
data <- cbind(data, socn_matrix)

# Step 2: (Quasi-Sequence-Order Descriptor)
qso_matrix <- t(sapply(data$pep0, extractQSO, nlag = 3))
data <- cbind(data, qso_matrix)

head(data[, c(grep("Schneider", names(data)), grep("Grantham", names(data)))])

# Part 5: Pseudo-amino acid composition
# Step 1: (PseAAC)
paac_matrix <- t(sapply(data$pep0, extractPAAC, lambda = 3, w = 0.05))
data <- cbind(data, paac_matrix)

# Step 2: (APseAAC)
apaac_matrix <- t(sapply(data$pep0, extractAPAAC, lambda = 3, w = 0.05))
data <- cbind(data, apaac_matrix)

# Part 6: Extract basic physicochemical properties
data <- data %>%
  mutate(
    net.charge = charge(pep0),                            # Net charge at pH=7
    hydrophobicity.value = hydrophobicity(pep0),          # Hydrophobicity index
    boman.index = boman(pep0)                             # Potential peptide interaction index
  )

# Use the aaComp function to extract amino acid composition features
# Define a function to process a single sequence
process_sequence <- function(seq) {
  comp <- aaComp(seq)
  result <- comp[[1]][, "Mole%"]
  return(result)
}

# Apply this function to all sequences
results <- do.call(rbind, lapply(data$pep0, process_sequence))

# Convert the results to a data frame
results <- as.data.frame(results)

# Add the results to the original data frame
data <- cbind(data, results)

# Save the results to Excel
write_xlsx(data, "C:/Users/zzh/Desktop/16w/Supplementary Data 9.xlsx")

