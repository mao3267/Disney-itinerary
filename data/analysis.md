# Touringplans Data Exploratory Analysis

## Dataset Overview

Source: `touringplans` R package (LucyMcGowan/touringplans)
14 ride datasets across 4 Walt Disney World parks + metadata.

## Ride Datasets Summary

Each ride CSV has 4 columns: `park_date`, `wait_datetime`, `wait_minutes_actual`, `wait_minutes_posted`.

| Ride | Rows | Date Range | Actual % | Posted % | Actual Dates | Posted Dates |
|------|------|-----------|----------|----------|-------------|-------------|
| pirates_of_caribbean | 301,946 | 2015-01-01 - 2021-12-28 | 3.84% | 88.90% | 2,118 | 2,199 |
| seven_dwarfs_train | 321,631 | 2015-01-01 - 2021-12-28 | 2.37% | 90.46% | 2,029 | 2,334 |
| splash_mountain | 287,948 | 2015-01-01 - 2021-12-28 | 2.39% | 86.49% | 1,819 | 2,144 |
| soarin | 274,770 | 2015-01-01 - 2021-12-28 | 3.22% | 94.85% | 2,007 | 2,184 |
| spaceship_earth | 277,248 | 2015-01-01 - 2021-12-28 | 1.62% | 93.76% | 1,743 | 2,328 |
| alien_saucers | 129,876 | 2018-06-30 - 2021-12-28 | 2.33% | 93.76% | 956 | 1,054 |
| rock_n_rollercoaster | 277,509 | 2015-01-01 - 2021-12-28 | 2.29% | 92.30% | 1,861 | 2,307 |
| slinky_dog | 135,946 | 2018-06-30 - 2021-12-28 | 4.42% | 87.74% | 1,007 | 1,054 |
| toy_story_mania | 284,170 | 2015-01-01 - 2021-12-28 | 3.69% | 92.94% | 2,073 | 2,318 |
| dinosaur | 252,403 | 2015-01-01 - 2021-12-28 | 2.00% | 90.58% | 1,842 | 2,214 |
| expedition_everest | 275,274 | 2015-01-01 - 2021-12-28 | 3.18% | 89.52% | 1,999 | 2,326 |
| flight_of_passage | 184,818 | 2017-05-26 - 2021-12-28 | 3.23% | 93.89% | 1,273 | 1,457 |
| kilimanjaro_safaris | 257,785 | 2015-01-01 - 2021-12-28 | 2.53% | 92.03% | 1,900 | 2,329 |
| navi_river | 182,121 | 2017-05-26 - 2021-12-28 | 2.51% | 93.33% | 1,304 | 1,454 |

### Key Findings: Data Availability

- **Posted wait times**: 87-95% non-null across all rides. ~1,000-2,300 unique dates with data.
- **Actual wait times**: Only 1.6-4.4% non-null. However, ~1,000-2,100 unique dates still have at least some actual observations.
- **Implication**: While per-observation actual data is very sparse, on a per-day basis actual data exists for ~75-95% of dates. After daily aggregation, actual-based statistics should be viable.

## Wait Time Statistics (Cleaned: 0 ≤ wait < 300 min)

| Ride | Posted Mean | Posted Std | Actual Mean | Actual Std | Over-prediction |
|------|------------|-----------|------------|-----------|----------------|
| pirates_of_caribbean | 28.5 | 18.0 | 18.2 | 13.0 | +10.3 |
| seven_dwarfs_train | 77.0 | 34.0 | 36.2 | 23.0 | +40.8 |
| splash_mountain | 43.7 | 30.3 | 25.4 | 18.7 | +18.3 |
| soarin | 45.7 | 27.3 | 25.0 | 15.8 | +20.7 |
| spaceship_earth | 18.8 | 14.6 | 9.8 | 9.3 | +9.0 |
| alien_saucers | 29.9 | 16.0 | 22.1 | 12.6 | +7.8 |
| rock_n_rollercoaster | 59.0 | 31.8 | 29.3 | 19.3 | +29.7 |
| slinky_dog | 72.7 | 27.8 | 40.7 | 22.7 | +32.0 |
| toy_story_mania | 54.2 | 29.9 | 26.4 | 18.3 | +27.8 |
| dinosaur | 27.4 | 19.6 | 20.2 | 15.3 | +7.2 |
| expedition_everest | 32.4 | 22.9 | 15.5 | 12.3 | +16.9 |
| flight_of_passage | 114.7 | 53.4 | 60.5 | 37.4 | +54.2 |
| kilimanjaro_safaris | 40.0 | 28.6 | 22.2 | 17.8 | +17.8 |
| navi_river | 62.5 | 32.2 | 29.0 | 22.4 | +33.5 |

### Key Findings: Over-prediction

- Disney consistently over-predicts wait times by **7-54 minutes** depending on the ride.
- Highest over-prediction: Flight of Passage (+54 min), Seven Dwarfs Train (+41 min), Navi River (+34 min).
- The ratio is not constant — it varies significantly by ride, so a simple correction factor won't work.

## Outlier / Data Quality Issues

| Ride | Issue | Details |
|------|-------|---------|
| seven_dwarfs_train | Extreme negative actual | Min = -92,918 (1 value, clearly erroneous) |
| toy_story_mania | Extreme positive actual | Max = 90,387 (2 values > 300) |
| flight_of_passage | Extreme positive actual | Max = 47,897 (9 values > 300) |
| flight_of_passage | Posted outliers | 100 posted values ≥ 300 (up to 390) |
| toy_story_mania | Posted outliers | 35 posted values ≥ 300 |

**Recommendation**: Filter wait times to [0, 300] range before aggregation.

## Cross-Ride Date Analysis

- **Common window (all 14 rides)**: 2018-06-30 to 2021-12-28 (~3.5 years)
- `alien_saucers` and `slinky_dog` start latest (2018-06-30)
- `flight_of_passage` and `navi_river` start 2017-05-26
- Remaining 10 rides start 2015-01-01

## Parks Metadata

- **Date range**: 2015-01-01 to 2021-08-31 (2,079 rows)
- Note: Metadata ends Aug 2021 but ride data extends to Dec 2021 (4 months gap)

### wdw_ticket_season (our planned season filter)

- Coverage: 1,218 / 2,079 dates (58.6%) — **significant gap**
- Distribution: peak=336, regular=615, value=267

### wdwseason (alternative)

- Coverage: 1,992 / 2,079 dates (95.8%) — much better coverage
- 17 categories (Christmas, Easter, Summer Break, Fall, etc.)

### Implication for Stage 2

`wdw_ticket_season` only covers 59% of dates. Options:
1. Use `wdwseason` instead (96% coverage) and map to peak/regular/value
2. Accept losing 41% of dates when filtering by season
3. Infer missing `wdw_ticket_season` from month/date patterns

## Attractions Metadata

| Name | Park | Duration (min) | Wait/100 guests |
|------|------|---------------|----------------|
| Pirates of Caribbean | Magic Kingdom | 7.5 | 1.5 |
| 7 Dwarfs Train | Magic Kingdom | 3.0 | 5.0 |
| Splash Mountain | Magic Kingdom | 18.0 | 3.5 |
| Soarin' | Epcot | 8.0 | 3.0 |
| Spaceship Earth | Epcot | 16.0 | 3.0 |
| Alien Saucers | Hollywood Studios | 2.5 | 10.0 |
| Rock Coaster | Hollywood Studios | 1.5 | 2.5 |
| Slinky Dog | Hollywood Studios | 3.0 | 5.0 |
| Toy Story Mania! | Hollywood Studios | 6.5 | 4.5 |
| DINOSAUR | Animal Kingdom | 3.5 | 3.0 |
| Expedition Everest | Animal Kingdom | 4.0 | 4.0 |
| Flight of Passage | Animal Kingdom | 6.0 | 4.0 |
| Kilimanjaro Safaris | Animal Kingdom | 20.0 | 4.0 |
| Na'vi River | Animal Kingdom | 5.0 | 5.0 |

## Stage 2 Implications

1. **Actual wait data is viable at daily granularity** — despite only 2-4% row-level fill rate, ~75-95% of dates have at least some actual observations per ride.
2. **Outlier filtering needed** — cap at [0, 300] min before aggregation.
3. **Season filter choice**: `wdw_ticket_season` has 41% missing dates. Should use `wdwseason` (96% coverage) mapped to 3 categories, or accept the gap.
4. **Common date window**: 2018-06-30 to 2021-08-31 (limited by metadata end date + latest ride start) for full 14-ride + season analysis.
5. **Metadata date gap**: Ride data goes to 2021-12-28 but parks_metadata ends 2021-08-31. Last 4 months of ride data won't have season labels.
