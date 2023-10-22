source("r/plots-common.R")

## justice: main dataset
## suppress encoding, due to MAIN_OFFENCE column
justice <- read.csv("data/justice-coded-2015-2022.csv.bz2",
  colClasses = "character"
)
## add earlier years
justice <- rbind(
  justice,
  read.csv("data/justice-coded-2010-2014.csv.bz2",
    colClasses = "character"
  )
)
justice <- rbind(
  justice,
  read.csv("data/justice-coded-2001-2009.csv.bz2",
    colClasses = "character"
  )
)

## make Value numeric
justice$Value <- as.numeric(justice$Value)

## only ANZSOC Divisions 01 -- 16
## See: https://datainfoplus.stats.govt.nz/item/nz.govt.stats/a10413bf-f78a-4f17-a9c1-55e7717ab91d
ja <- subset(justice, MAIN_OFFENCE %in%
  c(
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
    "11", "12", "13", "14", "15", "16"
  ))

## drop "Unknown" Ethnicity
ja <- subset(ja, Ethnicity != "Unknown")

## drop "No sentence recorded" Sentence
ja <- subset(ja, Sentence != "No sentence recorded")

## drop "Traffic and vehicle regulatory offences"
ja <- subset(ja, Main.offence != "Traffic and vehicle regulatory offences")

## also drop Monetary
## ja <- subset(ja, Sentence != "Monetary")

## stage 1 preprocessing
pp_justice_common <- function(df) {
  ## drop excess and coded columns
  for (c in c("Court", "Flags", "MAIN_OFFENCE", "COURT_CLUSTER", "AGE_GROUP", "GENDER", "ETHNICITY", "YEAR", "SENTENCE")) {
    df[, c] <- NULL
  }

  ## Rename one very long offence name
  df$Main.offence[df$Main.offence == "Offences against justice procedures, government security and government operations"] <- "Offences against justice procedures, gov. security and gov. operations"

  ## factorise columns of interest
  df$Main.offence <- as.factor(df$Main.offence)
  df$Ethnicity <- as.factor(df$Ethnicity)
  df$Sentence <- as.factor(df$Sentence)

  ## make binary label
  df$outcome <- !(df$Sentence %in% c("Imprisonment", "Imprisonment sentences", "Life Imprisonment"))

  ## expand rows based on Value count and remove the column
  df <- df[rep(rownames(df), df$Value), ]
  df$Value <- NULL

  ## drop offences if less than 100 observations
  ct_offence <- table(df$Main.offence)
  df <- df[!(df$Main.offence %in% names(
    ct_offence[ct_offence < 100]
  )), ]
  ## remake factor (to remove levels of dropped offences)
  df$Main.offence <- factor(df$Main.offence)

  df
}

## apply preprocessing
ja <- pp_justice_common(ja)

## specify the factored column
offence_column <- "Main.offence"
## compute base rates
X <- compute_disparate_impact(ja, offence_column)

## plot to file
title <- "Disparate impact of imprisonment sentences by ANZSOC division, "
years <- "2001-2022"
sub_note <- paste0("n=", format(nrow(ja), big.mark = ","))

do_plot <- function() {
  plot_disparate_impact(X, ja, offence_column, title, years,
    sub_note = sub_note,
    log = FALSE,
    cex_base = 0.8,
    cex_p = 0.01, title_at = 1,
    legend_position = "topright", legend_cex = 1.0
    )
}
svg("justice-disparate-base-2001-2022.svg", width = 11)
do_plot()
dev.off()

pdf("justice-disparate-base-2001-2022.pdf", width = 11)
do_plot()
dev.off()


##
