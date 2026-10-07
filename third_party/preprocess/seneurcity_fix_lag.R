#!/usr/bin/env Rscript

# Lag correction for filtered SenseurCity CSVs using the JRC R toolbox.
#
# Paper linkage:
# - docs/seneurcity.md recommends lag correction via Find_Max_CCF in
#   "151016 Sensor_Toolbox.R" (ec-jrc/airsenseur-calibration).
#
# This script is a thin CLI wrapper around that reference implementation.

suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(jsonlite))


parse_args <- function(argv) {
  args <- list(
    in_dir = "data/seneurcityCleaned",
    out_dir = "data/seneurcity_lag_fixed",
    pattern = "*.csv",
    sensors = c("OPCN3PM25", "5325CAT", "5325CST"),
    ref_col = "Ref.PM2.5",
    date_col = "date",
    lag_max = 120L,
    keep_na = FALSE,
    toolbox = "third_party/airsenseur-calibration/151016 Sensor_Toolbox.R"
  )

  i <- 1L
  while (i <= length(argv)) {
    tok <- argv[[i]]
    if (tok == "--keep-na") {
      args$keep_na <- TRUE
      i <- i + 1L
      next
    }

    if (tok %in% c("--in-dir", "--out-dir", "--pattern", "--ref-col", "--date-col", "--lag-max", "--toolbox", "--sensors")) {
      if (tok == "--sensors") {
        j <- i + 1L
        vals <- c()
        while (j <= length(argv) && !startsWith(argv[[j]], "--")) {
          vals <- c(vals, argv[[j]])
          j <- j + 1L
        }
        if (length(vals) == 0L) {
          stop("--sensors requires at least one value")
        }
        args$sensors <- vals
        i <- j
        next
      }

      if (i == length(argv)) {
        stop(paste0(tok, " requires a value"))
      }
      val <- argv[[i + 1L]]
      if (tok == "--in-dir") args$in_dir <- val
      if (tok == "--out-dir") args$out_dir <- val
      if (tok == "--pattern") args$pattern <- val
      if (tok == "--ref-col") args$ref_col <- val
      if (tok == "--date-col") args$date_col <- val
      if (tok == "--toolbox") args$toolbox <- val
      if (tok == "--lag-max") args$lag_max <- as.integer(val)
      i <- i + 2L
      next
    }

    stop(paste0("Unknown argument: ", tok))
  }

  return(args)
}


shift_by_ccf_lag <- function(vec, lag) {
  # Matches the toolbox's shifting logic (see DeLag_Cal in 151016 Sensor_Toolbox.R).
  n <- length(vec)
  lag <- as.integer(lag)
  if (is.na(lag) || lag == 0L || n == 0L) {
    return(vec)
  }
  if (lag > 0L) {
    return(c(vec[(lag + 1L):n], rep(NA, lag)))
  }
  return(c(rep(NA, -lag), vec[1L:(n - (-lag))]))
}


main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))

  if (!file.exists(args$toolbox)) {
    stop(paste0("Toolbox not found: ", args$toolbox))
  }
  source(args$toolbox)
  if (!exists("Find_Max_CCF")) {
    stop("Find_Max_CCF not found after sourcing toolbox")
  }

  in_dir <- normalizePath(args$in_dir, mustWork = TRUE)
  out_dir <- normalizePath(args$out_dir, mustWork = FALSE)
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  files <- sort(Sys.glob(file.path(in_dir, args$pattern)))
  if (length(files) == 0L) {
    stop(paste0("No files found in ", in_dir, " matching pattern ", args$pattern))
  }

  cat("Found", length(files), "files in", in_dir, "\n")
  cat("Output directory:", out_dir, "\n")

  summary <- list()

  for (f in files) {
    df <- fread(f, check.names = FALSE)
    rows_in <- nrow(df)

    # Some pre-filtered files can legitimately be empty; keep them as-is.
    if (rows_in == 0L) {
      out_path <- file.path(out_dir, basename(f))
      fwrite(df, out_path)
      summary[[basename(f)]] <- list(
        file = basename(f),
        skipped = TRUE,
        reason = "empty input file (0 rows)",
        rows_in = 0,
        rows_out = 0
      )
      cat("  SKIP", basename(f), ": empty input file (0 rows)\n")
      next
    }

    required <- c(args$date_col, args$ref_col, args$sensors)
    missing <- setdiff(required, names(df))
    if (length(missing) > 0L) {
      summary[[basename(f)]] <- list(
        file = basename(f),
        skipped = TRUE,
        reason = paste0("missing required columns: ", paste(missing, collapse = ", "))
      )
      cat("  SKIP", basename(f), ":", summary[[basename(f)]]$reason, "\n")
      next
    }

    # Ensure deterministic order.
    df[[args$date_col]] <- as.POSIXct(df[[args$date_col]], format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
    df <- df[order(df[[args$date_col]]), ]

    # Coerce numeric columns.
    df[[args$ref_col]] <- as.numeric(df[[args$ref_col]])
    for (s in args$sensors) {
      df[[s]] <- as.numeric(df[[s]])
    }

    lag_results <- list()
    for (s in args$sensors) {
      lag_res <- Find_Max_CCF(df[[s]], df[[args$ref_col]], Lag.max = as.integer(args$lag_max))
      lag_val <- as.integer(lag_res$lag)
      ccf_cor <- as.numeric(lag_res$cor)

      df[[s]] <- shift_by_ccf_lag(df[[s]], lag_val)
      corr0 <- suppressWarnings(cor(df[[s]], df[[args$ref_col]], use = "complete.obs"))

      lag_results[[s]] <- list(
        lag = lag_val,
        ccf_cor = ccf_cor,
        corr0_after = corr0
      )
    }

    rows_before_dropna <- nrow(df)
    if (!isTRUE(args$keep_na)) {
      df <- df[complete.cases(df[, c(args$date_col, args$ref_col, args$sensors), with = FALSE]), ]
    }
    rows_out <- nrow(df)

    # Convert date back to ISO8601 (UTC) so downstream stays consistent.
    df[[args$date_col]] <- strftime(df[[args$date_col]], "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")

    out_path <- file.path(out_dir, basename(f))
    fwrite(df, out_path)

    summary[[basename(f)]] <- list(
      file = basename(f),
      skipped = FALSE,
      rows_in = rows_in,
      rows_before_dropna = rows_before_dropna,
      rows_out = rows_out,
      rows_dropped = rows_before_dropna - rows_out,
      lag_results = lag_results
    )

    # Progress line
    parts <- c()
    for (s in args$sensors) {
      parts <- c(parts, sprintf("%s:lag=%d", s, lag_results[[s]]$lag))
    }
    cat(sprintf("  %s: %d -> %d rows | %s\n", basename(f), rows_in, rows_out, paste(parts, collapse = " | ")))
  }

  summary_payload <- list(
    config = list(
      in_dir = args$in_dir,
      out_dir = args$out_dir,
      pattern = args$pattern,
      sensors = args$sensors,
      ref_col = args$ref_col,
      date_col = args$date_col,
      lag_max = as.integer(args$lag_max),
      drop_na = !isTRUE(args$keep_na),
      toolbox = args$toolbox
    ),
    files = unname(summary)
  )

  summary_path <- file.path(out_dir, "lag_correction_summary.json")
  write_json(summary_payload, path = summary_path, pretty = TRUE, auto_unbox = TRUE)
  cat("Saved summary:", summary_path, "\n")
}


main()
