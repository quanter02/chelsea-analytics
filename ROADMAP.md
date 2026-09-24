# Roadmap

| Stage | What | Builds on |
|---|---|---|
| v1 (done) | Post-mortem + rule-based game plan, live season dossier, manager fingerprint, league translation | this repo |
| v2 | **LLM briefing**: export findings as JSON and have an LLM write a one-page coach briefing and half-time notes | `scout()` output |
| v3 | **Matchup model**: predict expected xG for "style A vs style B" from team style vectors (PPDA, long balls, counter xG…) | `season_team_matches()` (760 rows) |
| v4 | **Set-piece lab**: classify corner defending (zonal / man) from StatsBomb 360 freeze frames and find weak zones | `shots_table()` + freeze frames |
| v5 | **Pressing triggers** from open tracking data (SkillCorner) | new `tracking.py` |
| — | **J1 League 2024** (free Hudl StatsBomb data): run the same scouting on an Asian league | `data.py` |

Also planned: pre-match predictions and post-match grading for every Chelsea game, so the model's calls are checked in public.
