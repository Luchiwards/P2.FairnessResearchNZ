source("r/plots-common.R")

## police: main dataset
police <- read.csv("data/dataset_police.csv.bz2", stringsAsFactors = TRUE)

## drop Traffic
police <- subset(police, ANZSOC.Division != "Traffic and Vehicle Regulatory Offences")
police$ANZSOC.Division <- factor(police$ANZSOC.Division)

## convert Year.Month to Date
police$Year.Month <- as.Date(paste0("01-", police$Year.Month), "%d-%b-%Y")

## police2: police without Ethnicity == Not Stated
police2 <- police[police$Ethnicity != "Not Stated", ]

## stage 1 preprocessing
pp_police_common <- function(df) {
  ## drop Organisations
  df <- df[df$Person.Organisation == "Person", ]

  ## make binary label
  df$outcome <- df$Method.of.Proceeding != "Court Action"


  ## expand rows based on Proceedings count and remove the column
  df <- df[rep(rownames(df), df$Proceedings), ]
  df$Proceedings <- NULL
  df
}

## stage 2 preprocessing
pp_2_police_common <- function(df) {
  ## get counts of ethnicities
  counts <- table(df$Ethnicity)
  ## combine ethnicies < 10,000 into "Other"
  ## have to defactorise the column first
  df$Ethnicity <- as.character(df$Ethnicity)
  for (i in seq_along(counts)) {
    if (counts[i] < 10000) {
      df[df$Ethnicity == names(counts)[i], "Ethnicity"] <- "Other"
    }
  }
  ## back to factor
  df$Ethnicity <- as.factor(df$Ethnicity)
  df
}


police <- pp_police_common(police)
police2 <- pp_police_common(police2)


## view ethnicities
ethnicities <- table(police$Ethnicity)
ethnicities <- sort(ethnicities / sum(ethnicities))
pdf("ethnicities-proportion.pdf")
barplot(cbind(ethnicities),
  main = "Ethnicities",
  names.arg = "Proportion of Police dataset, descending order",
  las = 1,
  legend.text = TRUE,
  args.legend = list(x = "top", bty = "n"),
)
dev.off()

police <- pp_2_police_common(police)
police2 <- pp_2_police_common(police2)
offence_column <- "ANZSOC.Division"

X <- compute_disparate_impact(police, offence_column)
title <-
  "Disparate impact of Police referral to court action by ANZSOC Division, "
years <- "2014-2023"
pdf("police-disparate-base.pdf", width = 11)
plot_disparate_impact(X, police, offence_column, title, years,
  log = TRUE,
  legend_cex = 0.8
)
dev.off()

X <- compute_disparate_impact(police2, "ANZSOC.Division")

do_plot <- function() {
  plot_disparate_impact(X, police2, offence_column, title, years,
    sub_note = paste0("n=", format(nrow(police2), big.mark = ",")),
    title_at = 1,
    cex_base = 1,
    legend_position = "topright"
  )
}

svg("police2-disparate-base.svg", width = 11)
do_plot()
dev.off()

pdf("police2-disparate-base.pdf", width = 11)
do_plot()
dev.off()


## outliers
##
## Proportion of each offence represented by Ethnicity == Not Stated
not_stated <- table(police[police$Ethnicity == "Not Stated", "ANZSOC.Division"]) /
  table(police[, "ANZSOC.Division"])
##
pdf("not-stated-by-offence.pdf", width = 12)
old <- par(mar = c(4.1, 23, 4.1, 2.1))
barplot(
  horiz = TRUE, las = 1, cex.names = 0.8,
  main = "Share of observations, Not Stated",
  xlab =
    "count of observations with Ethnicity == Not Stated / number of observations",
  not_stated[order(not_stated)]
)
## abline(v = seq(0.01, 0.05, 0.01), lty = 3)
par(old)
dev.off()

## outliers
##
pdf("police-ethnicities.pdf")
ethnicities <- table(police$Ethnicity)
ethnicities <- sort(ethnicities / sum(ethnicities))
barplot(cbind(ethnicities),
  main = "Ethnicities",
  las = 1,
  legend.text = TRUE
)
dev.off()

## outliers
##
## These two groups had no favourable outcome, but the baseline rate
## is only 3%, and the sample size is small.
## print(X[X$disparate_impact == 0, ])

## outliers
##
## These offences had extremely low comparison rates, very poor
## disparate impact.
## print(X[X$disparate_impact < 0.5 & X$disparate_impact > 0, ])

##
## disparate impact < 0.80
## print(X[X$disparate_impact < 0.8, c("offence", "comparison_group", "disparate_impact")])

## print(X[, c("offence", "comparison_group", "disparate_impact")])

## try a facet plot
## library(ggplot2)
## ggplot(X, aes(disparate_impact, offence, col = comparison_group)) +
##   geom_point() +
##   theme_minimal()
