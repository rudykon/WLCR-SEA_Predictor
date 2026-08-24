#!/usr/bin/env Rscript

# Revision-9 bilingual main-manuscript figures for WLCR-SEA.
#
# This is intentionally independent of render_audit_wlcr_sea_figures.R.  It keeps
# Fig. 2 and Fig. 4 on their registered Revision-8 clean/audit evidence while
# replacing Fig. 3 with the Revision-9 missingness design:
#   (a) descriptive A6-versus-frozen-Original-WLCR stress curves; and
#   (b) matched-augmentation, paired cell-cluster forest evidence against
#       DLinear-Aug, PatchTST-Aug, and GRU-D-Direct-Aug.
#
# The script reads only declared artifact CSV/JSON files.  It never opens a
# traffic test input.  All plotted measurements must certify their source as
# the registered training trace before rendering begins.

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(scales)
  library(svglite)
  library(ragg)
  library(jsonlite)
})

COL <- c(
  green = "#005A18", red = "#A60000", brown = "#6B2C0C", black = "#000000",
  grey = "#595959", grey_light = "#F2F2F2", grid = "#C7C7C7", text = "#000000",
  teal = "#005A18", teal_light = "#F2F2F2", navy = "#000000", navy_light = "#F2F2F2",
  amber = "#6B2C0C", amber_light = "#F2F2F2", purple = "#A60000", purple_light = "#F2F2F2",
  grey_dark = "#000000", white = "#FFFFFF"
)

parse_args <- function(args) {
  out <- list(
    clean_analysis = "artifacts/reproduction/analysis/clean",
    audit = "artifacts/reproduction/analysis/audit",
    revision9 = "artifacts/reproduction/analysis/missingness",
    output = "artifacts/reproduction/figures",
    check_only = FALSE, audit_only = FALSE
  )
  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (identical(key, "--check-only")) {
      out$check_only <- TRUE
      i <- i + 1L
      next
    }
    if (identical(key, "--audit-only")) {
      out$audit_only <- TRUE
      i <- i + 1L
      next
    }
    if (!startsWith(key, "--") || i == length(args)) {
      stop("arguments must be supplied as --name value (or --check-only)")
    }
    name <- chartr("-", "_", sub("^--", "", key))
    if (!name %in% c("clean_analysis", "audit", "revision9", "output")) {
      stop("unknown argument: ", key)
    }
    out[[name]] <- args[[i + 1L]]
    i <- i + 2L
  }
  out
}

font_family <- function(lang) {
  if (identical(lang, "zh")) "Noto Serif CJK SC" else "Nimbus Roman"
}

pub_theme <- function(lang, base_size = 6.0) {
  theme_bw(base_family = font_family(lang), base_size = base_size) +
    theme(
      text = element_text(colour = COL[["text"]]),
      plot.background = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA),
      panel.border = element_rect(colour = COL[["black"]], fill = NA, linewidth = 0.28),
      panel.grid.major = element_line(colour = COL[["grid"]], linewidth = 0.18),
      panel.grid.minor = element_blank(),
      axis.line = element_blank(),
      axis.ticks = element_line(colour = COL[["black"]], linewidth = 0.26),
      axis.ticks.length = unit(1.8, "pt"),
      axis.title = element_text(size = 5.9, colour = COL[["black"]]),
      axis.text = element_text(size = 5.25, colour = COL[["black"]]),
      plot.title = element_text(size = 6.15, face = "plain", hjust = 0.5, margin = margin(b = 1.5)),
      plot.subtitle = element_text(size = 4.8, colour = COL[["grey"]], hjust = 0),
      plot.tag = element_text(size = 6.1, face = "bold"),
      strip.background = element_blank(),
      strip.text = element_text(size = 5.8, face = "plain", colour = COL[["black"]]),
      legend.position = "inside",
      legend.text = element_text(size = 4.65),
      legend.title = element_blank(),
      legend.key.height = unit(6.0, "pt"),
      legend.key.width = unit(10, "pt"),
      legend.background = element_rect(fill = "white", colour = COL[["black"]], linewidth = 0.24),
      legend.box.background = element_blank(),
      legend.key = element_rect(fill = "white", colour = NA),
      plot.margin = margin(2.5, 3.5, 2.5, 3.5)
    )
}

export_plot <- function(plot, stem, width, height, lang) {
  dir.create(dirname(stem), recursive = TRUE, showWarnings = FALSE)
  family <- font_family(lang)
  grDevices::cairo_pdf(
    paste0(stem, ".pdf"), width = width, height = height,
    family = family, bg = "white", onefile = FALSE
  )
  print(plot)
  grDevices::dev.off()
  svglite::svglite(
    paste0(stem, ".svg"), width = width, height = height,
    bg = "white", system_fonts = list(sans = family)
  )
  print(plot)
  grDevices::dev.off()
  ragg::agg_png(
    paste0(stem, ".png"), width = width, height = height,
    units = "in", res = 300, background = "white"
  )
  print(plot)
  grDevices::dev.off()
  ragg::agg_tiff(
    paste0(stem, ".tiff"), width = width, height = height,
    units = "in", res = 600, background = "white", compression = "lzw"
  )
  print(plot)
  grDevices::dev.off()
}

assert_equal_audit_axes_boxes <- function(stem) {
  # The four panel borders are emitted by ggplot2 as the only SVG rectangles
  # with this exact border style. Check the rendered geometry rather than
  # relying on patchwork's layout defaults, which previously allowed the two
  # nested rows to allocate different panel widths.
  svg_path <- paste0(stem, ".svg")
  svg <- paste(readLines(svg_path, warn = FALSE, encoding = "UTF-8"), collapse = "\n")
  pattern <- "<rect x='([0-9.]+)' y='([0-9.]+)' width='([0-9.]+)' height='([0-9.]+)' style='stroke-width: 0\\.60;'\\s*/>"
  matched <- regmatches(svg, gregexpr(pattern, svg, perl = TRUE))[[1]]
  if (length(matched) != 4L) {
    stop("expected exactly four audit panel borders in ", svg_path, "; found ", length(matched))
  }
  boxes <- do.call(rbind, lapply(matched, function(item) {
    fields <- regmatches(item, regexec(pattern, item, perl = TRUE))[[1]]
    as.numeric(fields[2:5])
  }))
  colnames(boxes) <- c("x", "y", "width", "height")
  tolerance_pt <- 0.02
  if (any(abs(boxes[, "width"] - boxes[1L, "width"]) > tolerance_pt) ||
      any(abs(boxes[, "height"] - boxes[1L, "height"]) > tolerance_pt)) {
    stop("audit axes boxes are not equal after rendering: ", svg_path)
  }
  invisible(boxes)
}

audit_legend_inside_theme <- function(x, y) {
  if (utils::packageVersion("ggplot2") >= "3.5.0") {
    theme(
      legend.position = "inside",
      legend.position.inside = c(x, y),
      legend.justification.inside = c(1, 1)
    )
  } else {
    theme(legend.position = c(x, y), legend.justification = c(1, 1))
  }
}

sha256_file <- function(path) {
  value <- system2("sha256sum", path, stdout = TRUE)
  if (length(value) != 1L || !nzchar(value[[1]])) stop("unable to hash ", path)
  strsplit(value[[1]], "[[:space:]]+")[[1]][[1]]
}

require_files <- function(paths, message) {
  absent <- paths[!file.exists(paths)]
  if (length(absent)) stop(message, ": ", paste(absent, collapse = ", "))
}

require_columns <- function(data, fields, label) {
  missing <- setdiff(fields, names(data))
  if (length(missing)) stop(label, " lacks required columns: ", paste(missing, collapse = ", "))
  invisible(data)
}

assert_finite <- function(values, label) {
  numeric <- suppressWarnings(as.numeric(unlist(values, use.names = FALSE)))
  if (!length(numeric) || any(!is.finite(numeric))) {
    stop(label, " has missing or non-finite numeric values")
  }
  invisible(values)
}

project_root <- function() normalizePath(".", mustWork = TRUE)

path_inside <- function(path, allowed_root, label) {
  root <- normalizePath(allowed_root, mustWork = TRUE)
  candidate <- normalizePath(path, mustWork = TRUE)
  prefix <- paste0(root, "/")
  if (!identical(candidate, root) && !startsWith(candidate, prefix)) {
    stop(label, " must remain inside ", root)
  }
  candidate
}

output_inside_reproduction <- function(path) {
  root <- project_root()
  candidate <- normalizePath(path, mustWork = FALSE)
  allowed <- normalizePath(file.path(root, "artifacts", "reproduction"), mustWork = TRUE)
  prefix <- paste0(allowed, "/")
  if (!startsWith(candidate, prefix)) stop("output must remain inside artifacts/reproduction/")
  candidate
}

relative_project_path <- function(path) {
  root <- project_root()
  resolved <- normalizePath(path, mustWork = TRUE)
  prefix <- paste0(root, "/")
  if (!startsWith(resolved, prefix)) stop("path is outside project root: ", resolved)
  substring(resolved, nchar(prefix) + 1L)
}

assert_final_test_unopened <- function(path, label) {
  payload <- fromJSON(path, simplifyVector = FALSE)
  value <- payload$finals_test_opened
  if (is.null(value) || !identical(as.logical(value), FALSE)) {
    stop(label, " does not certify finals_test_opened = false")
  }
  invisible(payload)
}

validate_manifest <- function(root, manifest_path, expected_relatives, label) {
  entries <- fromJSON(manifest_path, simplifyVector = FALSE)
  if (!is.list(entries) || !length(entries) || is.null(entries[[1L]]$path)) {
    stop(label, " manifest must be a non-empty list of file records")
  }
  record_paths <- vapply(entries, function(item) as.character(item$path), character(1))
  for (relative in expected_relatives) {
    matches <- which(record_paths == relative)
    if (length(matches) != 1L) stop(label, " manifest is missing ", relative)
    item <- entries[[matches[[1L]]]]
    candidate <- normalizePath(file.path(root, relative), mustWork = TRUE)
    root_norm <- normalizePath(root, mustWork = TRUE)
    if (!startsWith(candidate, paste0(root_norm, "/"))) {
      stop(label, " manifest path escapes its artifact root: ", relative)
    }
    if (is.null(item$sha256) || !identical(sha256_file(candidate), as.character(item$sha256))) {
      stop(label, " manifest SHA256 mismatch for ", relative)
    }
    if (is.null(item$size_bytes) || file.info(candidate)$size != as.numeric(item$size_bytes)) {
      stop(label, " manifest size mismatch for ", relative)
    }
  }
  invisible(entries)
}

assert_revision9_summary <- function(summary) {
  if (!identical(summary$registered_train_file, "data/train_data.csv")) {
    stop("Revision-9 summary does not declare the registered training trace")
  }
  before <- as.character(summary$registered_train_sha256_before)
  after <- as.character(summary$registered_train_sha256_after)
  if (!identical(before, after) || !grepl("^[0-9a-f]{64}$", after)) {
    stop("Revision-9 training-file SHA256 is missing, malformed, or changed during analysis")
  }
  if (!identical(as.logical(summary$finals_test_opened), FALSE)) {
    stop("Revision-9 summary does not certify finals_test_opened = false")
  }
  protocol <- summary$protocol
  if (is.null(protocol) || as.integer(protocol$scenario_count) != 17L ||
      !identical(as.integer(unlist(protocol$corruption_seeds)), 142:146)) {
    stop("Revision-9 summary does not certify the full 17-scenario/five-mask protocol")
  }
  bootstrap <- summary$bootstrap_protocol
  if (is.null(bootstrap) || as.integer(bootstrap$replicates) != 5000L ||
      !isTRUE(bootstrap$original_wlcr_excluded_from_matched_claims)) {
    stop("Revision-9 summary does not certify the matched 5,000-replicate bootstrap contract")
  }
  models <- summary$models
  if (is.null(models$grud_direct_aug) || is.null(models$original_wlcr) ||
      !identical(models$original_wlcr$comparison_class,
        "descriptive_clean_trained_not_augmentation_matched")) {
    stop("Revision-9 summary lacks the GRU-D control or Original-WLCR separation")
  }
  if (!isTRUE(summary$clean_replay$original_worker_clean_replay_verified)) {
    stop("Revision-9 summary does not certify Original-WLCR clean replay")
  }
  invisible(summary)
}

method_labels <- function(lang) {
  if (identical(lang, "zh")) {
    c(
      A0_fixed = "固定季节专家混合",
      A0_global_static = "全局指标权重",
      A0_horizon_indicator = "预测步-指标权重",
      A1_softmax = "动态 Softmax",
      A2_entmax = "动态 Entmax",
      A3_hard_mask = "A3 硬屏蔽",
      A4_reliability = "A4 可靠性",
      A5_residual = "A5 有界残差",
      A6_mixed_aug = "WLCR-SEA",
      original_wlcr = "仅话务 LightGBM（73维）",
      standard_stat = "标准统计 LGBM",
      dlinear_clean = "DLinear（干净训练）",
      dlinear_aug = "DLinear（混合增强）",
      patchtst_clean = "PatchTST（干净训练）",
      patchtst_aug = "PatchTST（混合增强）",
      grud_direct_aug = "GRU-D-Direct-Aug"
    )
  } else {
    c(
      A0_fixed = "Fixed seasonal mixture",
      A0_global_static = "Global indicator weights",
      A0_horizon_indicator = "Horizon-indicator weights",
      A1_softmax = "Dynamic Softmax",
      A2_entmax = "Dynamic Entmax",
      A3_hard_mask = "A3 hard mask",
      A4_reliability = "A4 reliability",
      A5_residual = "A5 bounded residual",
      A6_mixed_aug = "WLCR-SEA",
      original_wlcr = "Traffic-only LightGBM (73D)",
      standard_stat = "Standard-stat LGBM",
      dlinear_clean = "DLinear (clean)",
      dlinear_aug = "DLinear-Aug",
      patchtst_clean = "PatchTST (clean)",
      patchtst_aug = "PatchTST-Aug",
      grud_direct_aug = "GRU-D-Direct-Aug"
    )
  }
}

load_grud_clean_row <- function(revision9) {
  curves <- read.csv(file.path(revision9, "comparative_missingness.csv"), stringsAsFactors = FALSE)
  required <- c(
    "method", "label", "training_view", "comparison_class", "scenario",
    "requested_rate", "corruption_seed_count", "macro_wape"
  )
  require_columns(curves, required, "Revision-9 curve aggregate")
  row <- curves[
    curves$method == "grud_direct_aug" & curves$scenario == "clean" &
      as.numeric(curves$requested_rate) == 0,
    , drop = FALSE
  ]
  if (nrow(row) != 1L) stop("Revision-9 curve aggregate needs exactly one GRU-D clean row")
  if (!identical(as.character(row$training_view[[1L]]), "mixed-15pct") ||
      !identical(as.character(row$comparison_class[[1L]]), "augmentation_matched") ||
      as.integer(row$corruption_seed_count[[1L]]) != 5L) {
    stop("GRU-D clean row does not satisfy the matched 15%-augmentation contract")
  }
  assert_finite(row$macro_wape, "GRU-D clean macro WAPE")
  row
}

make_clean <- function(clean_analysis, revision9, output, lang) {
  zh <- identical(lang, "zh")
  labels <- method_labels(lang)
  clean <- read.csv(file.path(clean_analysis, "comparative_clean_accuracy.csv"), stringsAsFactors = FALSE)
  require_columns(clean, c("method", "label", "training_view", "macro_wape"), "Revision-8 clean table")
  grud <- load_grud_clean_row(revision9)
  clean_min <- clean[, c("method", "label", "training_view", "macro_wape"), drop = FALSE]
  grud_min <- data.frame(
    method = "grud_direct_aug",
    label = as.character(grud$label[[1L]]),
    training_view = as.character(grud$training_view[[1L]]),
    macro_wape = as.numeric(grud$macro_wape[[1L]]),
    stringsAsFactors = FALSE
  )
  clean_min <- rbind(clean_min, grud_min)

  strong <- c(
    "dlinear_clean", "patchtst_clean", "original_wlcr", "A6_mixed_aug",
    "dlinear_aug", "patchtst_aug", "grud_direct_aug", "standard_stat"
  )
  a <- clean_min[match(strong, clean_min$method), , drop = FALSE]
  if (any(is.na(a$method))) stop("clean comparison is missing a required strong baseline")
  assert_finite(a$macro_wape, "clean panel macro WAPE")
  a <- a[order(a$macro_wape), , drop = FALSE]
  a$display <- factor(labels[a$method], levels = rev(labels[a$method]))
  a$family <- ifelse(
    a$method == "A6_mixed_aug", "sea",
    ifelse(grepl("^dlinear", a$method), "dlinear",
      ifelse(grepl("^patchtst", a$method), "patchtst",
        ifelse(a$method == "grud_direct_aug", "grud", "reference")))
  )
  a$training <- ifelse(
    a$method %in% c("A6_mixed_aug", "dlinear_aug", "patchtst_aug", "grud_direct_aug"),
    "mixed", "clean"
  )
  a$marker <- ifelse(
    a$method == "A6_mixed_aug", "proposed",
    a$training
  )
  xmin <- floor(min(a$macro_wape) * 1000) / 1000 - 0.002
  p_a <- ggplot(a, aes(macro_wape, display)) +
    geom_segment(
      aes(x = xmin, xend = macro_wape, yend = display),
      colour = COL[["black"]], linewidth = 0.28
    ) +
    geom_point(
      aes(colour = family, shape = marker), fill = COL[["white"]],
      size = 1.75, stroke = 0.46
    ) +
    geom_text(
      aes(label = sprintf("%.4f", macro_wape)), hjust = -0.18,
      size = 1.62, family = font_family(lang)
    ) +
    scale_colour_manual(
      values = c(
        sea = COL[["green"]], dlinear = COL[["black"]],
        patchtst = COL[["brown"]], grud = COL[["red"]],
        reference = COL[["black"]]
      ), guide = "none"
    ) +
    scale_shape_manual(values = c(clean = 21, mixed = 25, historical = 22, proposed = 23), guide = "none") +
    scale_x_continuous(
      limits = c(xmin, max(a$macro_wape) + 0.014),
      labels = label_number(accuracy = 0.01)
    ) +
    labs(
      title = if (zh) "(a) 干净留出集准确性" else "(a) Clean holdout accuracy",
      x = if (zh) "宏平均 WAPE（越低越好）" else "Macro WAPE (lower is better)",
      y = NULL
    ) +
    pub_theme(lang) +
    theme(legend.position = "none")

  hierarchy <- c(
    "A0_fixed", "A0_global_static", "A0_horizon_indicator", "A1_softmax",
    "A2_entmax", "A3_hard_mask", "A4_reliability", "A5_residual",
    "A6_mixed_aug"
  )
  b <- clean[match(hierarchy, clean$method), , drop = FALSE]
  if (any(is.na(b$method))) stop("router hierarchy is incomplete")
  b$display <- factor(labels[b$method], levels = rev(labels[hierarchy]))
  b$stage <- ifelse(
    b$method == "A0_fixed", "fixed",
    ifelse(b$method %in% c("A0_global_static", "A0_horizon_indicator"),
      "indicator", ifelse(b$method == "A6_mixed_aug", "final", "request"))
  )
  bx <- floor(min(b$macro_wape) * 1000) / 1000 - 0.002
  p_b <- ggplot(b, aes(macro_wape, display)) +
    geom_segment(
      aes(x = bx, xend = macro_wape, yend = display),
      colour = COL[["black"]], linewidth = 0.28
    ) +
    geom_point(
      aes(colour = stage, shape = stage), fill = COL[["white"]], size = 1.72, stroke = 0.46
    ) +
    geom_text(
      aes(label = sprintf("%.4f", macro_wape)), hjust = -0.17,
      size = 1.60, family = font_family(lang)
    ) +
    scale_colour_manual(
      values = c(
        fixed = COL[["black"]], indicator = COL[["brown"]],
        request = COL[["red"]], final = COL[["green"]]
      ), guide = "none"
    ) +
    scale_shape_manual(values = c(fixed = 22, indicator = 24, request = 25, final = 23), guide = "none") +
    scale_x_continuous(
      limits = c(bx, max(b$macro_wape) + 0.017),
      labels = label_number(accuracy = 0.01)
    ) +
    labs(
      title = if (zh) "路由层级" else "Routing hierarchy",
      x = if (zh) "宏平均 WAPE" else "Macro WAPE", y = NULL
    ) +
    pub_theme(lang) +
    theme(legend.position = "none")

  combined <- p_b
  stem <- file.path(output, if (zh) "wlcr_sea_clean_accuracy_zh" else "wlcr_sea_clean_accuracy")
  export_plot(combined, stem, 3.50, 2.45, lang)
}

condition_labels <- function(lang) {
  if (identical(lang, "zh")) {
    c(
      block_20pct = "连续块 20%",
      timeline_tail_20pct = "时间轴尾段 20%",
      asynchronous_30pct = "异步 30%",
      block_50pct = "连续块 50%",
      timeline_tail_50pct = "时间轴尾段 50%",
      asynchronous_50pct = "异步 50%"
    )
  } else {
    c(
      block_20pct = "Block 20%",
      timeline_tail_20pct = "Timeline tail 20%",
      asynchronous_30pct = "Asynchronous 30%",
      block_50pct = "Block 50%",
      timeline_tail_50pct = "Timeline tail 50%",
      asynchronous_50pct = "Asynchronous 50%"
    )
  }
}

curve_method_labels <- function(lang) {
  if (identical(lang, "zh")) {
    c(
      A6_mixed_aug = "WLCR-SEA",
      original_wlcr = "原始 WLCR（冻结）"
    )
  } else {
    c(
      A6_mixed_aug = "WLCR-SEA",
      original_wlcr = "Original WLCR (frozen)"
    )
  }
}

mechanism_labels <- function(lang) {
  if (identical(lang, "zh")) {
    c(block = "连续块缺失", recent_tail = "时间轴尾段缺失", asynchronous = "指标异步缺失")
  } else {
    c(block = "Block missingness", recent_tail = "Timeline-tail missingness", asynchronous = "Asynchronous missingness")
  }
}

prepare_original_stress_curves <- function(revision9, lang) {
  all_curves <- read.csv(file.path(revision9, "comparative_missingness.csv"), stringsAsFactors = FALSE)
  required <- c(
    "method", "training_view", "comparison_class", "scenario", "mechanism",
    "requested_rate", "corruption_seed_count", "macro_wape", "macro_wape_ci_low",
    "macro_wape_ci_high"
  )
  require_columns(all_curves, required, "Revision-9 curve aggregate")
  keep <- c("A6_mixed_aug", "original_wlcr")
  mechanisms <- c("block", "recent_tail", "asynchronous")
  clean <- all_curves[
    all_curves$method %in% keep & all_curves$scenario == "clean" &
      as.numeric(all_curves$requested_rate) == 0,
    , drop = FALSE
  ]
  clean <- clean[!duplicated(clean$method), , drop = FALSE]
  if (nrow(clean) != length(keep)) {
    stop("Revision-9 curve aggregate must provide one shared clean replay per displayed method")
  }
  curves <- all_curves[
    all_curves$method %in% keep & all_curves$mechanism %in% mechanisms &
      as.numeric(all_curves$requested_rate) > 0,
    , drop = FALSE
  ]
  expected <- length(keep) * length(mechanisms) * 4L
  if (nrow(curves) != expected) stop("Revision-9 displayed stress curves are incomplete")
  clean_repeated <- do.call(rbind, lapply(mechanisms, function(mechanism) {
    item <- clean
    item$mechanism <- mechanism
    item
  }))
  curves <- rbind(curves, clean_repeated)
  if (any(as.integer(curves$corruption_seed_count) != 5L)) {
    stop("Revision-9 stress curves do not aggregate exactly five fixed corruption masks")
  }
  a6 <- curves[curves$method == "A6_mixed_aug", , drop = FALSE]
  original <- curves[curves$method == "original_wlcr", , drop = FALSE]
  if (!all(a6$comparison_class == "augmentation_matched") ||
      !all(a6$training_view == "mixed-15pct")) {
    stop("WLCR-SEA stress curve is not labelled as the matched augmented estimate")
  }
  if (!all(original$comparison_class == "descriptive_clean_trained_not_augmentation_matched") ||
      !all(original$training_view == "clean-trained frozen model")) {
    stop("Original-WLCR stress curve is not labelled as descriptive clean-trained evidence")
  }
  assert_finite(curves[, c("macro_wape", "macro_wape_ci_low", "macro_wape_ci_high")], "stress curves")
  if (any(curves$macro_wape_ci_low > curves$macro_wape) ||
      any(curves$macro_wape_ci_high < curves$macro_wape)) {
    stop("stress-curve confidence interval does not contain its point estimate")
  }
  labels <- curve_method_labels(lang)
  mechanisms_display <- mechanism_labels(lang)
  curves$method_display <- factor(curves$method, levels = names(labels), labels = labels)
  curves$mechanism_display <- factor(
    curves$mechanism, levels = names(mechanisms_display), labels = mechanisms_display
  )
  curves <- curves[order(curves$mechanism, curves$method, curves$requested_rate), , drop = FALSE]
  curves
}

prepare_matched_forest <- function(revision9, lang) {
  forest <- read.csv(file.path(revision9, "paired_cell_bootstrap.csv"), stringsAsFactors = FALSE)
  required <- c(
    "condition", "baseline", "comparison_class", "bootstrap_replicates",
    "delta_a6_minus_baseline", "ci_low", "ci_high"
  )
  require_columns(forest, required, "Revision-9 matched bootstrap table")
  baseline_keys <- c("dlinear_aug", "patchtst_aug", "grud_direct_aug")
  conditions <- names(condition_labels(lang))
  forest <- forest[forest$baseline %in% baseline_keys, , drop = FALSE]
  if (nrow(forest) != length(baseline_keys) * length(conditions) ||
      anyDuplicated(forest[, c("baseline", "condition")])) {
    stop("Revision-9 matched bootstrap table must contain three baselines across six conditions")
  }
  if (!setequal(unique(forest$condition), conditions) ||
      !all(forest$comparison_class == "augmentation_matched") ||
      any(as.integer(forest$bootstrap_replicates) != 5000L)) {
    stop("Revision-9 matched bootstrap rows violate the fixed comparison contract")
  }
  assert_finite(forest[, c("delta_a6_minus_baseline", "ci_low", "ci_high")], "matched bootstrap table")
  if (any(forest$ci_low > forest$delta_a6_minus_baseline) ||
      any(forest$ci_high < forest$delta_a6_minus_baseline)) {
    stop("matched bootstrap interval does not contain its point estimate")
  }
  baseline_display <- if (identical(lang, "zh")) {
    c(dlinear_aug = "DLinear-Aug", patchtst_aug = "PatchTST-Aug", grud_direct_aug = "GRU-D-Direct-Aug")
  } else {
    c(dlinear_aug = "DLinear-Aug", patchtst_aug = "PatchTST-Aug", grud_direct_aug = "GRU-D-Direct-Aug")
  }
  labels <- condition_labels(lang)
  forest$baseline_display <- factor(
    forest$baseline, levels = baseline_keys, labels = baseline_display[baseline_keys]
  )
  forest$condition_display <- factor(
    labels[forest$condition], levels = rev(labels[conditions])
  )
  forest$ci_excludes_zero <- forest$ci_high < 0 | forest$ci_low > 0
  forest
}

validate_original_descriptive_bootstrap <- function(revision9) {
  descriptive <- read.csv(
    file.path(revision9, "paired_cell_bootstrap_original_wlcr_descriptive.csv"),
    stringsAsFactors = FALSE
  )
  require_columns(
    descriptive,
    c("condition", "baseline", "comparison_class", "bootstrap_replicates"),
    "Revision-9 Original-WLCR descriptive bootstrap table"
  )
  if (nrow(descriptive) != 6L || !all(descriptive$baseline == "original_wlcr") ||
      !all(descriptive$comparison_class == "descriptive_clean_trained_not_augmentation_matched") ||
      any(as.integer(descriptive$bootstrap_replicates) != 5000L)) {
    stop("Original-WLCR bootstrap table is not a six-condition descriptive-only record")
  }
  invisible(descriptive)
}

make_missingness <- function(revision9, output, lang) {
  zh <- identical(lang, "zh")
  curves <- prepare_original_stress_curves(revision9, lang)
  forest <- prepare_matched_forest(revision9, lang)
  validate_original_descriptive_bootstrap(revision9)
  curve_labels <- curve_method_labels(lang)
  palette <- setNames(c(COL[["teal"]], COL[["grey_dark"]]), curve_labels)
  shapes <- setNames(c(23, 22), curve_labels)
  linetypes <- setNames(c("solid", "longdash"), curve_labels)

  p_a <- ggplot(
    curves,
    aes(
      requested_rate, macro_wape, colour = method_display,
      shape = method_display, linetype = method_display, group = method_display
    )
  ) +
    geom_errorbar(
      aes(ymin = macro_wape_ci_low, ymax = macro_wape_ci_high),
      width = 0.012, linewidth = 0.25, show.legend = FALSE
    ) +
    geom_line(linewidth = 0.42) +
    geom_point(size = 1.35, stroke = 0.40, fill = COL[["white"]]) +
    facet_wrap(~mechanism_display, nrow = 1) +
    scale_colour_manual(values = palette) +
    scale_shape_manual(values = shapes) +
    scale_linetype_manual(values = linetypes) +
    scale_x_continuous(
      breaks = c(0, 0.1, 0.2, 0.3, 0.5), labels = label_percent(accuracy = 1)
    ) +
    labs(
      title = if (zh) "(a) 缺失压力曲线" else "(a) Missingness stress curves",
      x = if (zh) "请求破坏率" else "Requested corruption",
      y = if (zh) "宏平均 WAPE" else "Macro WAPE"
    ) +
    pub_theme(lang) +
    audit_legend_inside_theme(0.02, 0.98) +
    theme(
      legend.direction = "vertical",
      legend.key.width = unit(10, "pt"),
      legend.spacing.y = unit(1, "pt"),
      strip.text = element_text(size = 5.7, face = "plain")
    )

  baseline_levels <- levels(forest$baseline_display)
  baseline_colours <- setNames(
    c(COL[["navy"]], COL[["amber"]], COL[["purple"]]), baseline_levels
  )
  limits <- range(c(forest$ci_low, forest$ci_high, 0), finite = TRUE)
  span <- diff(limits)
  padding <- max(0.0015, span * 0.08)
  p_b <- ggplot(forest, aes(y = condition_display)) +
    geom_vline(
      xintercept = 0, colour = COL[["black"]], linewidth = 0.28, linetype = "dashed"
    ) +
    geom_segment(
      aes(
        x = ci_low, xend = ci_high, yend = condition_display,
        colour = baseline_display
      ), linewidth = 0.38
    ) +
    geom_point(
      aes(
        x = delta_a6_minus_baseline, colour = baseline_display, shape = baseline_display
      ), fill = COL[["white"]], size = 1.38, stroke = 0.42
    ) +
    facet_wrap(~baseline_display, nrow = 1) +
    scale_colour_manual(values = baseline_colours, guide = "none") +
    scale_shape_manual(values = setNames(c(22, 24, 25), baseline_levels), guide = "none") +
    scale_x_continuous(
      limits = c(limits[[1L]] - padding, limits[[2L]] + padding),
      labels = label_number(accuracy = 0.01)
    ) +
    labs(
      title = if (zh) "(b) 增强匹配的配对差值" else "(b) Augmentation-matched paired differences",
      x = if (zh) "WLCR-SEA − 基线的宏平均 WAPE 差值" else "Macro-WAPE difference: WLCR-SEA − baseline",
      y = NULL
    ) +
    pub_theme(lang) +
    theme(
      strip.text = element_text(size = 5.7, face = "plain"),
      panel.spacing.x = unit(5.0, "pt"),
      axis.text.y = element_text(size = 5.15),
      legend.position = "none",
      plot.margin = margin(2.5, 3.5, 2.5, 3.5)
    )

  combined <- p_a / p_b + plot_layout(heights = c(1.00, 1.00))
  stem <- file.path(output, if (zh) "wlcr_sea_missingness_zh" else "wlcr_sea_missingness")
  export_plot(combined, stem, 7.16, 3.60, lang)
}

interval_row <- function(summary, field, label) {
  item <- summary$seed_intervals[[field]]
  if (is.null(item)) stop("full audit summary is missing ", field)
  data.frame(
    key = field,
    label = label,
    mean = as.numeric(item$mean),
    low = as.numeric(item$ci_low),
    high = as.numeric(item$ci_high),
    stringsAsFactors = FALSE
  )
}

short_expert_labels <- function(lang) {
  if (identical(lang, "zh")) {
    c(
      last_day = "前一日", last_week = "前一周", last_biweek = "前两周",
      same_hour_median_7d = "7日中位", same_hour_median_14d = "14日中位",
      bounded_week_trend = "周趋势", window_local_median = "窗口中位"
    )
  } else {
    c(
      last_day = "D-1", last_week = "W-1", last_biweek = "W-2",
      same_hour_median_7d = "Median-7", same_hour_median_14d = "Median-14",
      bounded_week_trend = "Trend", window_local_median = "Window"
    )
  }
}

make_audit <- function(audit, output, lang) {
  zh <- identical(lang, "zh")
  deletion <- fromJSON(file.path(audit, "deletion_bootstrap.json"), simplifyVector = FALSE)
  summary <- fromJSON(file.path(audit, "summary.json"), simplifyVector = FALSE)
  invariance <- fromJSON(file.path(audit, "request_local_invariance.json"), simplifyVector = FALSE)
  targets <- fromJSON(file.path(audit, "request_local_targets.json"), simplifyVector = TRUE)
  loo <- read.csv(file.path(audit, "leave_one_out_summary.csv"), stringsAsFactors = FALSE)
  if (!isTRUE(invariance$bitwise_request_local_invariance_pass) ||
      as.integer(invariance$n_violations) != 0L) {
    stop("request-local invariance audit is not a zero-violation pass")
  }

  top <- deletion$top_deletion_minus_original
  random <- deletion$expected_matched_random_deletion_minus_original
  d <- data.frame(
    condition = if (zh) c("匹配随机删", "最高权重删") else c("MR del.", "TW del."),
    delta = c(as.numeric(random$delta_expected_random_minus_original), as.numeric(top$delta_proposed_minus_baseline)),
    low = c(as.numeric(random$ci_low), as.numeric(top$ci_low)),
    high = c(as.numeric(random$ci_high), as.numeric(top$ci_high))
  )
  d$condition <- factor(d$condition, levels = rev(d$condition))
  p_a <- ggplot(d, aes(delta, condition)) +
    geom_errorbarh(aes(xmin = low, xmax = high), height = 0.15, linewidth = 0.30, colour = COL[["black"]]) +
    geom_point(aes(colour = condition, shape = condition), fill = COL[["white"]], size = 1.48, stroke = 0.42) +
    geom_text(aes(label = sprintf("%+.4f", delta)), hjust = -0.25, size = 1.62, family = font_family(lang)) +
    scale_colour_manual(values = setNames(c(COL[["green"]], COL[["black"]]), levels(d$condition)), guide = "none") +
    scale_shape_manual(values = setNames(c(23, 22), levels(d$condition)), guide = "none") +
    scale_x_continuous(limits = c(0, max(d$high) * 1.30)) +
    labs(
      title = if (zh) "(a) 删除忠实度" else "(a) Deletion fidelity",
      x = if (zh) "相对原预测的宏 WAPE 增量" else "Macro-WAPE increase from original", y = NULL
    ) +
    pub_theme(lang) + theme(legend.position = "none", axis.text.y = element_text(angle = 30, hjust = 1, vjust = 0.5))

  td <- as.data.frame(targets$targets, stringsAsFactors = FALSE)
  require_columns(td, c("target_date", "missingness_bin"), "request-local target list")
  bins <- c("none_0pct", "low_0_to_10pct", "moderate_10_to_25pct", "high_above_25pct")
  bin_labels <- c(none_0pct = "0%", low_0_to_10pct = "0–10%", moderate_10_to_25pct = "10–25%", high_above_25pct = ">25%")
  dates <- sort(unique(as.character(td$target_date)))
  grid <- expand.grid(target_date = dates, missingness_bin = bins, stringsAsFactors = FALSE, KEEP.OUT.ATTRS = FALSE)
  counts <- aggregate(rep(1L, nrow(td)), by = list(target_date = as.character(td$target_date), missingness_bin = as.character(td$missingness_bin)), FUN = sum)
  names(counts)[3] <- "n"
  coverage <- merge(grid, counts, by = c("target_date", "missingness_bin"), all.x = TRUE)
  coverage$n[is.na(coverage$n)] <- 0L
  coverage$target_date <- factor(coverage$target_date, levels = dates)
  coverage$missingness_bin <- factor(
    coverage$missingness_bin, levels = bins, labels = unname(bin_labels[bins])
  )
  coverage_palette <- setNames(
    c(COL[["black"]], COL[["red"]], COL[["brown"]], COL[["green"]]),
    unname(bin_labels[bins])
  )
  coverage_shapes <- setNames(c(22, 25, 24, 23), unname(bin_labels[bins]))
  coverage_linetypes <- setNames(c("longdash", "solid", "solid", "solid"), unname(bin_labels[bins]))
  date_display <- if (zh) sub("2024-08-", "8/", dates) else sub("2024-08-", "Aug ", dates)
  p_b <- ggplot(
    coverage,
    aes(
      target_date, n, colour = missingness_bin, shape = missingness_bin,
      linetype = missingness_bin, group = missingness_bin
    )
  ) +
    geom_line(linewidth = 0.40) +
    geom_point(fill = COL[["white"]], size = 1.32, stroke = 0.40) +
    scale_colour_manual(values = coverage_palette) +
    scale_shape_manual(values = coverage_shapes) +
    scale_linetype_manual(values = coverage_linetypes) +
    scale_x_discrete(labels = date_display) +
    scale_y_continuous(limits = c(0, 30), breaks = seq(0, 30, 5), expand = expansion(mult = c(0, 0))) +
    labs(
      title = if (zh) "(b) 请求局部性覆盖" else "(b) Request-locality coverage",
      x = NULL, y = if (zh) "请求数" else "Requests"
    ) +
    pub_theme(lang) +
    audit_legend_inside_theme(0.97, 0.97) +
    theme(
      legend.direction = "vertical",
      legend.key.width = unit(10, "pt"),
      axis.text.x = element_text(size = 5.2),
      axis.title.y = element_text(margin = margin(r = if (zh) -18 else -22))
    )

  expert_labels <- short_expert_labels(lang)
  if (!all(loo$expert %in% names(expert_labels))) stop("leave-one-out table contains an unexpected expert")
  loo$display <- expert_labels[loo$expert]
  label_nudge <- max(loo$mean_absolute_prediction_change, na.rm = TRUE) * 0.055
  p_c <- ggplot(loo, aes(mean_attention_when_available, mean_absolute_prediction_change)) +
    geom_errorbar(aes(ymin = mean_absolute_prediction_change_ci_low, ymax = mean_absolute_prediction_change_ci_high), width = 0.004, colour = COL[["black"]], linewidth = 0.28) +
    geom_point(aes(size = availability_rate), shape = 23, fill = COL[["white"]], colour = COL[["green"]], stroke = 0.44) +
    geom_text(aes(label = display), nudge_y = label_nudge, size = 1.45, family = font_family(lang), check_overlap = TRUE) +
    scale_size(range = c(1.15, 1.75), guide = "none") +
    labs(
      title = if (zh) "(c) 专家影响" else "(c) Expert influence",
      x = if (zh) "可用时平均注意力质量" else "Mean attention mass when available",
      y = if (zh) "平均绝对预测变化" else "Mean absolute prediction change"
    ) +
    pub_theme(lang) + theme(
      legend.position = "none",
      axis.title.y = element_text(margin = margin(r = if (zh) -18 else -22))
    )

  restraint <- rbind(
    interval_row(summary, "mean_prior_mass", if (zh) "先验质量" else "Prior mass"),
    interval_row(summary, "residual_ratio_p50", if (zh) "残差 P50" else "Res. P50"),
    interval_row(summary, "residual_ratio_p90", if (zh) "残差 P90" else "Res. P90"),
    interval_row(summary, "hard_availability_max_unavailable_weight", if (zh) "不可用权重" else "Unavail. wt.")
  )
  envelope <- as.numeric(summary$maximum_bounded_envelope_violation)
  restraint <- rbind(restraint, data.frame(key = "bounded_envelope_violation", label = if (zh) "包络违例" else "Env. viol.", mean = envelope, low = envelope, high = envelope, stringsAsFactors = FALSE))
  restraint$label <- factor(restraint$label, levels = rev(restraint$label))
  support <- summary$seed_intervals$effective_support_mean
  if (is.null(support)) stop("full audit summary is missing effective support")
  support_text <- if (zh) sprintf("有效支持 %.2f/8；局部性：通过", as.numeric(support$mean)) else sprintf("Effective support %.2f/8; request locality: PASS", as.numeric(support$mean))
  xmax <- max(c(restraint$high, restraint$mean), na.rm = TRUE)
  xmax <- max(xmax * 1.16, 0.01)
  p_d <- ggplot(restraint, aes(mean, label)) +
    geom_segment(aes(x = 0, xend = mean, yend = label), colour = COL[["black"]], linewidth = 0.28) +
    geom_errorbarh(aes(xmin = pmax(0, low), xmax = high), height = 0.12, linewidth = 0.28, colour = COL[["black"]]) +
    geom_point(shape = 23, fill = COL[["white"]], colour = COL[["green"]], size = 1.48, stroke = 0.44) +
    geom_text(aes(label = percent(mean, accuracy = 0.01)), hjust = -0.20, size = 1.52, family = font_family(lang)) +
    scale_x_continuous(limits = c(0, xmax), labels = label_percent(accuracy = 1)) +
    labs(title = if (zh) "(d) 结构性检查" else "(d) Structural checks", x = NULL, y = NULL) +
    pub_theme(lang) + theme(legend.position = "none", axis.text.y = element_text(angle = 30, hjust = 1, vjust = 0.5))

  # Use one layout tree for all four panels. Combining two independent rows
  # made patchwork reserve unequal space for axis labels and produced wider
  # axes boxes on the top row than on the bottom row.
  combined <- wrap_plots(p_a, p_b, p_c, p_d, ncol = 2) +
    plot_layout(widths = c(1, 1), heights = c(1, 1))
  stem <- file.path(output, if (zh) "wlcr_sea_auditability_zh" else "wlcr_sea_auditability")
  export_plot(combined, stem, 7.16, 3.45, lang)
  assert_equal_audit_axes_boxes(stem)
}

hash_records <- function(paths) {
  result <- lapply(paths, sha256_file)
  names(result) <- vapply(paths, relative_project_path, character(1))
  result
}

build_qa <- function(args, required, files, revision9_summary) {
  list(
    schema_version = 9,
    backend = "R ggplot2 + patchwork + cairo_pdf + svglite + ragg",
    vector_formats = c("pdf", "svg"),
    raster_formats = c("png_300dpi", "tiff_600dpi_lzw"),
    bilingual = TRUE,
    final_canvas_inches = list(
      figure_2 = list(width = 3.50, height = 2.45),
      figure_3 = list(width = 7.16, height = 3.60),
      figure_4 = list(width = 7.16, height = 3.45)
    ),
    evidence_status = "exploratory_redesign_on_existing_trace",
    finals_test_opened = FALSE,
    provenance = list(
      registered_train_file = "data/train_data.csv",
      registered_train_sha256 = as.character(revision9_summary$registered_train_sha256_after),
      input_scope = "Revision-8 clean/audit artifacts and Revision-9 training-trace-derived missingness artifacts only",
      revision9_protocol = "17 scenarios; five fixed global corruption masks (142--146)",
      request_identity_alignment = "certified by Revision-9 Original-WLCR artifact ingestion before aggregation",
      per_mask_hashes = "certified by Revision-9 input manifest and summary",
      original_wlcr_status = "frozen clean-trained descriptive stress control; excluded from augmentation-matched superiority inference"
    ),
    figure_contract = list(
      figure_2 = list(
        core_conclusion = "Clean accuracy preserves the routing hierarchy and includes GRU-D-Direct-Aug as a 15%-mixed-augmentation neural control.",
        panels = list(a = "Clean-holdout point estimates including GRU-D-Direct-Aug.", b = "Fixed-to-learned routing hierarchy.")
      ),
      figure_3 = list(
        core_conclusion = "Matched augmentation robustness is inferred only from the three paired neural controls; Original WLCR is shown only as a clean-trained descriptive stress curve.",
        archetype = "asymmetric quantitative grid",
        panels = list(
          a = "Three aligned mechanism facets: A6 and frozen clean-trained Original WLCR-LightGBM, explicitly not augmentation matched.",
          b = "Full-width grouped forest: A6 minus DLinear-Aug, PatchTST-Aug, and GRU-D-Direct-Aug across six fixed listed conditions."
        )
      ),
      figure_4 = list(core_conclusion = "Request-local auditability remains supported by the registered Revision-8 audit.")
    ),
    statistics = list(
      curves = "Five fixed corruption-mask t intervals; not model-seed uncertainty.",
      forest = "5,000-replicate unadjusted paired cell-cluster percentile 95% intervals after fixed-mask paired deltas are averaged inside each replicate.",
      original_wlcr = "Separate descriptive bootstrap record is validated but intentionally not plotted as matched-inference evidence."
    ),
    inputs = hash_records(required),
    outputs = hash_records(files)
  )
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  clean_root <- path_inside(args$clean_analysis, "artifacts/reproduction", "clean_analysis")
  audit_root <- path_inside(args$audit, "artifacts/reproduction", "audit")
  revision9_root <- path_inside(args$revision9, "artifacts/reproduction", "revision9")
  output <- output_inside_reproduction(args$output)
  clean_required <- c(
    file.path(clean_root, "comparative_clean_accuracy.csv"),
    file.path(clean_root, "summary.json"),
    file.path(clean_root, "manifest.json")
  )
  audit_required <- c(
    file.path(audit_root, "deletion_bootstrap.json"),
    file.path(audit_root, "leave_one_out_summary.csv"),
    file.path(audit_root, "request_local_invariance.json"),
    file.path(audit_root, "request_local_targets.json"),
    file.path(audit_root, "summary.json"),
    file.path(audit_root, "manifest.json")
  )
  revision9_required <- c(
    file.path(revision9_root, "comparative_missingness.csv"),
    file.path(revision9_root, "paired_cell_bootstrap.csv"),
    file.path(revision9_root, "paired_cell_bootstrap_original_wlcr_descriptive.csv"),
    file.path(revision9_root, "summary.json"),
    file.path(revision9_root, "manifest.json")
  )
  required <- c(clean_required, audit_required, revision9_required)
  require_files(required, "missing Revision-9 figure inputs")
  clean_summary <- assert_final_test_unopened(file.path(clean_root, "summary.json"), "Revision-8 clean analysis")
  audit_summary <- assert_final_test_unopened(file.path(audit_root, "summary.json"), "Revision-8 audit")
  revision9_summary <- assert_final_test_unopened(file.path(revision9_root, "summary.json"), "Revision-9 analysis")
  assert_revision9_summary(revision9_summary)
  validate_manifest(clean_root, file.path(clean_root, "manifest.json"), c("comparative_clean_accuracy.csv", "summary.json"), "Revision-8 clean")
  validate_manifest(audit_root, file.path(audit_root, "manifest.json"), c("deletion_bootstrap.json", "leave_one_out_summary.csv", "request_local_invariance.json", "request_local_targets.json", "summary.json"), "Revision-8 audit")
  validate_manifest(revision9_root, file.path(revision9_root, "manifest.json"), c("comparative_missingness.csv", "paired_cell_bootstrap.csv", "paired_cell_bootstrap_original_wlcr_descriptive.csv", "summary.json"), "Revision-9")
  if (!identical(clean_summary$evidence_status, "exploratory_redesign_on_existing_trace") ||
      !identical(audit_summary$evidence_status, "exploratory_redesign_on_existing_trace")) {
    stop("Revision-8 clean/audit inputs do not carry the expected evidence status")
  }
  if (isTRUE(args$check_only)) {
    message("Revision-9 figure inputs passed validation; --check-only emitted no files.")
    return(invisible(NULL))
  }
  if (isTRUE(args$audit_only)) {
    for (lang in c("en", "zh")) make_audit(audit_root, output, lang)
  } else {
    for (lang in c("en", "zh")) {
      make_clean(clean_root, revision9_root, output, lang)
      make_missingness(revision9_root, output, lang)
      make_audit(audit_root, output, lang)
    }
  }
  stems <- c(
    "wlcr_sea_clean_accuracy", "wlcr_sea_clean_accuracy_zh",
    "wlcr_sea_missingness", "wlcr_sea_missingness_zh",
    "wlcr_sea_auditability", "wlcr_sea_auditability_zh"
  )
  formats <- c("pdf", "svg", "png", "tiff")
  files <- unlist(lapply(stems, function(stem) file.path(output, paste0(stem, ".", formats))))
  require_files(files, "one or more Revision-9 figure exports are missing")
  qa <- build_qa(args, required, files, revision9_summary)
  writeLines(
    jsonlite::toJSON(qa, auto_unbox = TRUE, pretty = TRUE),
    file.path(output, "revision9_figures_r_qa.json")
  )
}

main()
