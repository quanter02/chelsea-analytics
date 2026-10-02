# Data

Nothing in this folder is committed except this file. Everything is either downloaded on first run
or rebuilt on your own machine.

| Folder | Source | How you get it | Used by |
|---|---|---|---|
| `cache/` | [StatsBomb Open Data](https://github.com/statsbomb/open-data), PL 2015/16 | Downloaded automatically on first run (~100 MB, gzip) | `cfc.report` |
| `derived/` | Built from `cache/` | Rebuilt automatically | `cfc.report` |
| `understat_2026/` | [Understat](https://understat.com), EPL 2026/27 | `python -m cfc.live --fetch` | `cfc.live` |
| `translation/` | Understat player-seasons, 6 leagues, 2014/15–2025/26 | Collected manually (format below) | `cfc.translation_report` |
| `disclosure/` | DART Open API | `python -m disclosure.alert` writes daily alerts here | `disclosure.alert` |
| `trends/` | YouTube Data API, your own datasets | `python -m trends.collect ...` | `trends.run` |
| `fingerprint/` | Understat team-seasons, Bundesliga 2023/24 | Collected manually (format below) | `cfc.live` (manager fingerprint) |

**Why the Understat files aren't in the repo.** Understat data is free for personal and research use,
but it isn't mine to redistribute. The code is public, and the data stays on your own machine.

## Snapshot formats

The snapshots are compact text files: records separated by `;`, fields by `,`.

**`understat_2026/`**
- `matches.txt`: `match_id,date,home,away,home_score,away_score`
- `shots_1.txt`, `shots_2.txt`: `match_index,side(0=h/1=a),minute,X*1000,Y*1000,xG*10000,result,situation,body,last_action_code,player_idx,assist_idx`
  - result `G/W/S/M/B/P` = goal / own goal / saved / missed / blocked / post
  - situation `O/C/S/F/P` = open play / corner / set piece / direct free kick / penalty
  - body `R/L/H/X`
- `names.txt`: player names separated by `|`, indexed by `player_idx`
- `lastaction.json`: `{"Cross": "1", ...}`, the Understat last-action name → code
- `teams.txt`: `team,date,h_a,ppda_att,ppda_def,ppda_allowed_att,ppda_allowed_def,deep,deep_allowed,xpts`

**`translation/`** (per-90 values × 1000)
- `movers_*.txt`: `player,pos,from_to(e.g. DE),season(yy),min1,min2,npxG1,xA1,xGChain1,npxG2,xA2,xGChain2`
  - league codes `E` EPL, `S` La Liga, `D` Bundesliga, `I` Serie A, `F` Ligue 1, `R` Russian PL
- `stayers.json`: sufficient statistics per metric (`n, w, x, xx, xy, y, yy`) for players who stayed in the same league
- `candidates_2025.txt`: `player,pos,league,team,minutes,npxG,xA,xGChain,goals,assists`
- `epl_2025.txt`: `player,pos,npxG,xA,xGChain`

**`fingerprint/bundesliga_2023.txt`**:
`team,matches,xg,xga,npxg,npxga,ppda_att,ppda_def,ppda_allowed_att,ppda_allowed_def,deep,deep_allowed,xpts,pts,gf,ga`
