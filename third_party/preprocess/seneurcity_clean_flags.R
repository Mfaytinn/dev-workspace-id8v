#!/usr/bin/env Rscript

# Clean SenseurCity CSVs using pre-computed QA/QC flags.
#
# The raw SenseurCity CSVs ship with *_flag columns produced by the JRC toolbox
# (Functions4ASE.R / Filter_Sensor_Data). This script applies the paper-backed
# semantics:
#   - empty flag => valid
#   - non-empty flag => invalid => set value to NA
# then drops rows missing any required variables.
#
# Special case:
#   Antwerp_4043A7.csv is missing OPCN3PM25_flag. For missing OPCN3PM25_flag,
#   this script now computes PM2.5 outlier/range flags with the official JRC
#   toolbox function My.rm.Outliers() from Functions4ASE.R (no hard-range-only
#   fallback). If toolbox-based computation cannot run, the script fails.

suppressPackageStartupMessages(library(data.table))


parse_args <- function(argv) {
  args <- list(
    in_dir = "data/seneurcity",
    out_dir = "data/seneurcity_flags_cleaned",
    pattern = "Antwerp_*.csv",
    toolbox = "third_party/airsenseur-calibration/Functions4ASE.R"
  )

  i <- 1L
  while (i <= length(argv)) {
    tok <- argv[[i]]
    if (!(tok %in% c("--in-dir", "--out-dir", "--pattern", "--toolbox"))) {
      stop(paste0("Unknown argument: ", tok))
    }
    if (i == length(argv)) {
      stop(paste0(tok, " requires a value"))
    }
    val <- argv[[i + 1L]]
    if (tok == "--in-dir") args$in_dir <- val
    if (tok == "--out-dir") args$out_dir <- val
    if (tok == "--pattern") args$pattern <- val
    if (tok == "--toolbox") args$toolbox <- val
    i <- i + 2L
  }

  return(args)
}


KEEP_COLS <- c(
  "date",
  "latitude",
  "longitude",
  "SHT31TE",
  "SHT31HE",
  "Absolute_humidity",
  "Td_deficit",
  "BMP280",
  "OPCN3PM25",
  "5325CAT",
  "5325CST",
  "Ref.PM2.5"
)

REQUIRED_COLS <- KEEP_COLS

FLAG_MAP <- list(
  OPCN3PM25 = "OPCN3PM25_flag",
  `5325CAT` = "5325CAT_flag",
  `5325CST` = "5325CST_flag",
  BMP280 = "BMP280_flag",
  SHT31HE = "SHT31HE_flag",
  SHT31TE = "SHT31TE_flag"
)


trim_flag <- function(x) {
  if (is.null(x)) return(character())
  x <- as.character(x)
  x[is.na(x)] <- ""
  x <- trimws(x)
  return(x)
}


tokens_from_flag <- function(flag_str, allowed_tokens) {
  if (is.na(flag_str) || nchar(flag_str) == 0L) return(character())
  parts <- unlist(strsplit(flag_str, ",", fixed = TRUE), use.names = FALSE)
  parts <- trimws(parts)
  parts <- parts[nchar(parts) > 0L]
  parts <- parts[parts %in% allowed_tokens]
  return(unique(parts))
}


join_tokens <- function(tokens, order) {
  if (length(tokens) == 0L) return("")
  tokens <- unique(tokens)
  tokens <- tokens[order(match(tokens, order))]
  tokens <- tokens[!is.na(tokens)]
  if (length(tokens) == 0L) return("")
  return(paste(tokens, collapse = ","))
}


load_outlier_toolbox <- function(toolbox_path) {
  if (!requireNamespace("caTools", quietly = TRUE)) {
    stop("Required R package 'caTools' is not installed; cannot run toolbox outlier flagging.")
  }
  suppressPackageStartupMessages(library(caTools))
  if (!file.exists(toolbox_path)) {
    stop(paste0("Toolbox not found: ", toolbox_path))
  }
  source(toolbox_path)
  if (!exists("My.rm.Outliers")) {
    stop("My.rm.Outliers not found after sourcing toolbox")
  }
}


derive_quality_tokens <- function(flag_str) {
  # W/TRH/Inv/Ind are sensor-state flags; reuse from sister OPC channels.
  common_tokens <- c(
    "W",
    "T.min",
    "T.max",
    "Rh.min",
    "Rh.max",
    "Inv",
    "Ind"
  )
  return(tokens_from_flag(flag_str, common_tokens))
}


compute_missing_opcn3pm25_flag <- function(dt, toolbox_path) {
  load_outlier_toolbox(toolbox_path)

  if (!("date" %in% names(dt)) || !("OPCN3PM25" %in% names(dt))) {
    stop("Missing required columns for OPCN3PM25 flag computation: date, OPCN3PM25")
  }

  # Use a sibling OPC flag only for W/TRH/Inv token transfer.
  candidates <- c("OPCN3PM10_flag", "OPCN3PM1_flag")
  src_flag_col <- NULL
  for (c in candidates) {
    if (c %in% names(dt)) {
      src_flag_col <- c
      break
    }
  }
  if (is.null(src_flag_col)) {
    stop(
      "OPCN3PM25_flag is missing and no OPCN3PM10_flag/OPCN3PM1_flag exists ",
      "to transfer W/TRH/Inv tokens."
    )
  }

  date_vec <- as.POSIXct(dt[["date"]], format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  if (any(is.na(date_vec))) {
    # Fallback parse for non-ISO rows.
    date_vec <- as.POSIXct(dt[["date"]], tz = "UTC")
  }
  if (any(is.na(date_vec))) {
    stop("Failed to parse date for toolbox outlier computation")
  }

  y <- suppressWarnings(as.numeric(dt[["OPCN3PM25"]]))
  outli <- My.rm.Outliers(
    date = date_vec,
    y = y,
    ymin = 0,          # Table 11 PM min
    ymax = 300,        # Table 11 PM max
    ThresholdMin = 0,  # PM lower bound floor
    window = 181,      # Table 11 window
    threshold = 20,    # Table 11 threshold for PM
    plotting = FALSE,
    set.Outliers = TRUE
  )
  if (!all(c("Low_values", "High_values", "OutliersMin", "OutliersMax") %in% names(outli))) {
    stop("Toolbox outlier computation did not return expected columns")
  }
  if (nrow(outli) != nrow(dt)) {
    stop("Toolbox outlier computation returned unexpected row count")
  }

  token_order <- c(
    "Low_values",
    "High_values",
    "OutliersMin",
    "OutliersMax",
    "W",
    "T.min",
    "T.max",
    "Rh.min",
    "Rh.max",
    "Inv",
    "Ind"
  )

  src_flags <- trim_flag(dt[[src_flag_col]])

  out <- character(nrow(dt))
  for (i in seq_len(nrow(dt))) {
    toks <- character()
    if (isTRUE(outli$Low_values[i])) toks <- c(toks, "Low_values")
    if (isTRUE(outli$High_values[i])) toks <- c(toks, "High_values")
    if (isTRUE(outli$OutliersMin[i])) toks <- c(toks, "OutliersMin")
    if (isTRUE(outli$OutliersMax[i])) toks <- c(toks, "OutliersMax")
    toks <- c(toks, derive_quality_tokens(src_flags[i]))
    out[i] <- join_tokens(toks, token_order)
  }

  return(out)
}


clean_one <- function(src, dst, toolbox_path) {
  dt <- fread(src, check.names = FALSE)
  n_raw <- nrow(dt)

  missing <- setdiff(KEEP_COLS, names(dt))
  if (length(missing) > 0L) {
    cat("  SKIP", basename(src), ": missing columns", paste(missing, collapse = ", "), "\n")
    return(list(file = basename(src), skipped = TRUE, reason = paste(missing, collapse = ", ")))
  }

  # Repair missing OPCN3PM25_flag if needed.
  if (!("OPCN3PM25_flag" %in% names(dt))) {
    cat("  NOTE", basename(src), ": missing OPCN3PM25_flag; computing with Functions4ASE::My.rm.Outliers\n")
    dt[["OPCN3PM25_flag"]] <- compute_missing_opcn3pm25_flag(dt, toolbox_path)
  }

  # Apply flag-based masking.
  flag_counts <- list()
  for (col in names(FLAG_MAP)) {
    flag_col <- FLAG_MAP[[col]]
    if (!(flag_col %in% names(dt))) {
      next
    }
    flags <- trim_flag(dt[[flag_col]])
    flagged <- flags != ""
    flag_counts[[col]] <- as.integer(sum(flagged))
    set(dt, i = which(flagged), j = col, value = NA_real_)
  }

  dt_clean <- dt[, ..KEEP_COLS]
  dt_clean <- dt_clean[complete.cases(dt_clean[, ..REQUIRED_COLS])]

  dir.create(dirname(dst), showWarnings = FALSE, recursive = TRUE)
  fwrite(dt_clean, dst)

  n_clean <- nrow(dt_clean)
  dropped <- n_raw - n_clean
  pct <- if (n_raw > 0) (100.0 * dropped / n_raw) else 0.0
  cat(sprintf("  %s: %d -> %d (dropped %d, %.2f%%)\n", basename(src), n_raw, n_clean, dropped, pct))

  return(list(
    file = basename(src),
    skipped = FALSE,
    rows_raw = n_raw,
    rows_clean = n_clean,
    rows_dropped = dropped,
    flag_counts = flag_counts
  ))
}


main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  in_dir <- normalizePath(args$in_dir, mustWork = TRUE)
  out_dir <- normalizePath(args$out_dir, mustWork = FALSE)
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  files <- sort(Sys.glob(file.path(in_dir, args$pattern)))
  if (length(files) == 0L) {
    stop(paste0("No files found in ", in_dir, " matching ", args$pattern))
  }

  cat("Found", length(files), "CSV files in", in_dir, "\n")
  cat("Output directory:", out_dir, "\n\n")

  stats <- list()
  for (f in files) {
    stats[[basename(f)]] <- clean_one(
      f,
      file.path(out_dir, basename(f)),
      args$toolbox
    )
  }

  processed <- stats[!vapply(stats, function(s) isTRUE(s$skipped), logical(1))]
  total_raw <- sum(vapply(processed, function(s) s$rows_raw, numeric(1)))
  total_clean <- sum(vapply(processed, function(s) s$rows_clean, numeric(1)))

  cat("\n", paste(rep("=", 60), collapse = ""), "\n", sep = "")
  cat("Summary\n")
  cat(paste(rep("=", 60), collapse = ""), "\n", sep = "")
  cat("  Files processed:", length(processed), "\n")
  cat("  Total rows (raw):  ", format(total_raw, big.mark = ","), "\n")
  cat("  Total rows (clean):", format(total_clean, big.mark = ","), "\n")
  cat("  Total dropped:     ", format(total_raw - total_clean, big.mark = ","), "\n")
}


main()
