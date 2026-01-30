import datetime

from metamon.backend.team_prediction.usage_stats import get_usage_stats

FORMAT = "gen9ou"
START_DATE = datetime.date(2025, 1, 1)
END_DATE = datetime.date(2025, 6, 1)
RANK = 1500

stats = get_usage_stats(FORMAT, START_DATE, END_DATE, rank=RANK)

print("format", FORMAT)
print("date_window", START_DATE, END_DATE)
print("rank_requested", RANK)
print("rank_used", stats.rank)
print("pokemon_count", len(stats.usage))

# Show a small sample so we can sanity-check distributions.
top_n = 5
for mon in stats.usage[:top_n]:
    data = stats[mon]
    top_moves = sorted(data.get("moves", {}).items(), key=lambda x: x[1], reverse=True)[:3]
    top_items = sorted(data.get("items", {}).items(), key=lambda x: x[1], reverse=True)[:3]
    top_abilities = sorted(data.get("abilities", {}).items(), key=lambda x: x[1], reverse=True)[:3]
    print("\n", mon)
    print("  count", data.get("count"))
    print("  top_moves", top_moves)
    print("  top_items", top_items)
    print("  top_abilities", top_abilities)
