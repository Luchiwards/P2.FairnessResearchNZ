##
## compute_disparate_impact
##
## df:             dataframe
## offence_column: character string name of column which contains
##                 ANZSOC offence name
##
## returns:        dataframe with one row per offence level, and
##                 columns containing computed base rates
##
compute_disparate_impact <- function(df, offence_column) {
  ## define baseline and comparison groups
  baseline_group <- "European"
  groups <- levels(df$Ethnicity)

  ## sort offences by number of observations
  offences <- names(sort(table(df[, offence_column])))

  ## make matrix of offences by group
  X <- expand.grid(
    offence = offences,
    baseline_group = baseline_group,
    comparison_group = groups,
    general_rate = 0,
    baseline_rate = 0,
    comparison_rate = 0,
    disparate_impact = 0
  )

  for (off in offences) {
    ##
    subset <- df[df[, offence_column] == off, ]
    ##
    general_rate <- with(
      subset,
      mean(outcome == TRUE)
    )
    ##
    baseline_rate <- with(
      subset[subset$Ethnicity == baseline_group, ],
      mean(outcome == TRUE)
    )
    ##
    ## for (comparison_group in groups[-which(groups == baseline_group)]) {
    for (comparison_group in groups) {
      comparison_rate <- with(
        subset[subset$Ethnicity == comparison_group, ],
        mean(outcome == TRUE)
      )
      if (is.na(comparison_rate)) comparison_rate <- 0
      ##
      s <- X$offence == off &
        X$baseline_group == baseline_group &
        X$comparison_group == comparison_group
      X[s, "general_rate"] <- general_rate
      X[s, "baseline_rate"] <- baseline_rate
      X[s, "comparison_rate"] <- comparison_rate
      X[s, "disparate_impact"] <- comparison_rate / baseline_rate
    }
  }
  X
}

##
## plot_disparate_impact
##
## Given the output of compute_disparate_impact, generate a plot.
##
## X:  output of compute_disparate_impact
## df: original dataframe passed to compute_disparate_impact
## offence_column: character name of column containing ANZSOC offence
##                 levels.
## title: Base title of plot
## years: character string to append to plot title
## sub_note: character string to append to sub note
## log: to request log scale; will append "log scale" to title
## legend_position: passed to legend()'s legend.position
##
plot_disparate_impact <- function(X, df, offence_column, title, years,
                                  sub_note = "",
                                  log = FALSE, legend_position = "topright",
                                  cex_base = 0.5, cex_p = 0.01,
                                  legend_cex = 1.0,
                                  title_at = 0.5) {
  counts_by_ethnicity <- table(
    df[, offence_column],
    df$Ethnicity
  )

  x_scale <- function(x) if (log) log1p(x) else x
  ##
  ## assign specific symbols to specific ethnicities
  eth_code <- function(eth) {
    sapply(eth, function(x) {
      switch(as.character(x),
        "European" = 0,
        "Maori" = 1,
        "Pacific Island" = 2,
        "Pacific Peoples" = 2,
        "Asian" = 3,
        "Other" = 4,
        "Indian" = 5,
        "Not Stated" = 6,
        "Not Elsewhere Classified" = 7,
        "Middle Eastern" = 8,
        "African" = 9,
        "Latin American/Hispanic" = 10,
        11
      )
    })
  }

  ## adjustment parameters
  eth_code_col_adj <- 2
  ##
  ## make room for long offence names
  save_par <- par(mar = c(5.1, 22, 5.1, 2.1))
  ##
  plot.new()
  plot.window(xlim = range(x_scale(X$disparate_impact)), ylim = c(1, nlevels(X$offence)))
  ## make symbol size proportional to number of samples of that ethnicity
  for (off in X$offence) {
    sub <- subset(X, offence == off)
    cex <- cex_base + cex_p * sqrt(counts_by_ethnicity[
      off,
      sub$comparison_group
    ])
    points(offence ~ x_scale(disparate_impact),
      data = sub,
      pch = eth_code(sub$comparison_group),
      col = eth_code(sub$comparison_group) + eth_code_col_adj,
      cex = cex,
      xpd = TRUE
    )
  }
  ## abline(v = x_scale(0.8), col = 1, lty = 3)
  abline(v = x_scale(1), col = 1, lty = 2)
  ## abline(v = x_scale(1.2), col = 1, lty = 3)
  ticks <- seq(0, 2, by = 0.1)
  axis(1, at = x_scale(ticks), labels = ticks)
  mtext(levels(X$offence),
    side = 2, at = 1:nlevels(X$offence), las = 1,
    line = -1,
    cex = 0.8
  )
  title()
  mtext(
    paste0(
      title,
      ifelse(log, "log scale, ", ""), years
    ),
    side = 3,
    line = 1,
    font = 2, # bold
    at = title_at,
  )
  mtext(
    "P(. | C) / P(. | B), C is comparison group, B is base group (European)",
    side = 1,
    line = 3,
    at = title_at
  )
  mtext(
    paste("Symbol size proportional to population size", sub_note),
    side = 1,
    line = 4,
    at = title_at,
    cex = 0.8
  )
  legend(
    legend_position,
    legend = levels(X$comparison_group),
    pch = eth_code(levels(X$comparison_group)),
    col = eth_code(levels(X$comparison_group)) + eth_code_col_adj,
    cex = legend_cex
  )
  par(save_par)
}
