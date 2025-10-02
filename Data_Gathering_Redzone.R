library(nflfastR)
library(dplyr)

#Historic data
pbp <- load_pbp(2015:2023)

redzone_dtdp <- pbp %>%
  filter(yardline_100 <= 20, !is.na(down)) %>%
  select(
    play_id, game_id, old_game_id, season, week, season_type, game_date,
    home_team, away_team, posteam, posteam_type, defteam, side_of_field,
    drive, qtr, down, goal_to_go, ydstogo, yardline_100, time,
    quarter_seconds_remaining, half_seconds_remaining, game_seconds_remaining,
    game_half, quarter_end,
    play_type, desc, yards_gained, touchdown,
    td_team, td_player_name, td_player_id,
    shotgun, no_huddle, qb_dropback, qb_scramble, qb_spike, qb_kneel,
    pass_length, pass_location, air_yards, yards_after_catch,
    run_location, run_gap,
    sack, interception, fumble_lost,
    field_goal_result, kick_distance,
    two_point_attempt, two_point_conv_result, extra_point_result,
    total_home_score, total_away_score,
    posteam_score, defteam_score,
    score_differential, posteam_score_post, defteam_score_post, score_differential_post,
    home_timeouts_remaining, away_timeouts_remaining,
    posteam_timeouts_remaining, defteam_timeouts_remaining,
    epa, ep,
    wp, def_wp, home_wp, away_wp, wpa, vegas_wpa, vegas_home_wpa,
    no_score_prob, opp_fg_prob, opp_td_prob, opp_safety_prob,
    fg_prob, safety_prob, td_prob, extra_point_prob, two_point_conversion_prob,
    total_home_epa, total_away_epa,
    total_home_pass_epa, total_away_pass_epa,
    total_home_rush_epa, total_away_rush_epa,
    air_epa, yac_epa, comp_air_epa, comp_yac_epa
  )

write.csv(redzone_dtdp, "redzone_DTDP_2015_2023.csv", row.names = FALSE)


#2024 Data
pbp_2024 <- load_pbp(2024)

redzone_dtdp_2024 <- pbp_2024 %>%
  filter(yardline_100 <= 20, !is.na(down)) %>%
  select(
    play_id, game_id, old_game_id, season, week, season_type, game_date,
    home_team, away_team, posteam, posteam_type, defteam, side_of_field,
    drive, qtr, down, goal_to_go, ydstogo, yardline_100, time,
    quarter_seconds_remaining, half_seconds_remaining, game_seconds_remaining,
    game_half, quarter_end,
    play_type, desc, yards_gained, touchdown,
    td_team, td_player_name, td_player_id,
    shotgun, no_huddle, qb_dropback, qb_scramble, qb_spike, qb_kneel,
    pass_length, pass_location, air_yards, yards_after_catch,
    run_location, run_gap,
    sack, interception, fumble_lost,
    field_goal_result, kick_distance,
    two_point_attempt, two_point_conv_result, extra_point_result,
    total_home_score, total_away_score,
    posteam_score, defteam_score,
    score_differential, posteam_score_post, defteam_score_post, score_differential_post,
    home_timeouts_remaining, away_timeouts_remaining,
    posteam_timeouts_remaining, defteam_timeouts_remaining,
    epa, ep,
    wp, def_wp, home_wp, away_wp, wpa, vegas_wpa, vegas_home_wpa,
    no_score_prob, opp_fg_prob, opp_td_prob, opp_safety_prob,
    fg_prob, safety_prob, td_prob, extra_point_prob, two_point_conversion_prob,
    total_home_epa, total_away_epa,
    total_home_pass_epa, total_away_pass_epa,
    total_home_rush_epa, total_away_rush_epa,
    air_epa, yac_epa, comp_air_epa, comp_yac_epa
  )

write.csv(redzone_dtdp_2024, "redzone_DTDP_2024.csv", row.names = FALSE)
