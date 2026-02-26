library(touringplans)

output_dir <- file.path(getwd(), "data", "raw")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# All 14 ride datasets
rides <- c(
  # Magic Kingdom
  "pirates_of_caribbean", "seven_dwarfs_train", "splash_mountain",
  # Epcot
  "soarin", "spaceship_earth",
  # Hollywood Studios
  "alien_saucers", "rock_n_rollercoaster", "slinky_dog", "toy_story_mania",
  # Animal Kingdom
  "dinosaur", "expedition_everest", "flight_of_passage",
  "kilimanjaro_safaris", "navi_river"
)

for (ride_name in rides) {
  d <- get(ride_name)
  out_path <- file.path(output_dir, paste0(ride_name, ".csv"))
  write.csv(d, out_path, row.names = FALSE)
  cat(sprintf("Exported %s: %d rows -> %s\n", ride_name, nrow(d), out_path))
}

# Parks metadata (select relevant columns)
pm <- parks_metadata_raw
# Get column names that exist
available_cols <- colnames(pm)
wanted_cols <- c("date", "wdw_ticket_season", "wdwseason",
                 "wdwmaxtemp", "wdwmintemp", "wdwmeantemp",
                 "mkopen", "mkclose", "mkhours",
                 "epopen", "epclose", "ephours",
                 "hsopen", "hsclose", "hshours",
                 "akopen", "akclose", "akhours")
use_cols <- intersect(wanted_cols, available_cols)
pm_subset <- pm[, use_cols, drop = FALSE]
write.csv(pm_subset, file.path(output_dir, "parks_metadata.csv"), row.names = FALSE)
cat(sprintf("Exported parks_metadata: %d rows, %d cols\n", nrow(pm_subset), ncol(pm_subset)))

# Attractions metadata
write.csv(attractions_metadata, file.path(output_dir, "attractions_metadata.csv"), row.names = FALSE)
cat(sprintf("Exported attractions_metadata: %d rows\n", nrow(attractions_metadata)))

cat("\nDone. All files in:", output_dir, "\n")
