import os
import json
import datetime
from pathlib import Path

if "METAMON_CACHE_DIR" not in os.environ:
    os.environ["METAMON_CACHE_DIR"] = "/tmp/usage_stats_test"

from metamon.backend.team_prediction.usage_stats import get_usage_stats

base = Path('/tmp/usage_stats_test/usage-stats')
movesets_base = base / 'movesets_data' / 'gen9'
checks_base = base / 'checks_data' / 'gen9'

for p in [
    movesets_base / 'ou' / '1630',
    movesets_base / 'ou' / '1500',
    movesets_base / 'all_tiers' / '1630',
    movesets_base / 'all_tiers' / '1500',
    checks_base
]:
    p.mkdir(parents=True, exist_ok=True)

month = '2025-01'
json.dump(
    {'pikachu': {'count': 1, 'moves': {'Nothing': 1.0}}},
    open(movesets_base / 'ou' / '1630' / f'{month}.json', 'w'),
)
json.dump(
    {'pikachu': {'count': 1, 'moves': {'Thunderbolt': 1.0}}},
    open(movesets_base / 'ou' / '1500' / f'{month}.json', 'w'),
)
json.dump(
    {'pikachu': {'count': 1, 'moves': {'Iron Tail': 1.0}}},
    open(movesets_base / 'all_tiers' / '1630' / f'{month}.json', 'w'),
)
json.dump(
    {'pikachu': {'count': 1, 'moves': {'Surf': 1.0}}},
    open(movesets_base / 'all_tiers' / '1500' / f'{month}.json', 'w'),
)

stats = get_usage_stats('gen9ou', datetime.date(2025,1,1), datetime.date(2025,1,1), rank=1649)
print("rank_used", stats.rank)              # expect 1630 (nearest lower)
print("pikachu_moves", stats['pikachu']['moves'])  # expect Thunderbolt (lower-rank tier, before all_tiers)
