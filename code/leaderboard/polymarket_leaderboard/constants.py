"""Documented values and endpoints for the public Polymarket APIs."""

LEADERBOARD_URL = "https://data-api.polymarket.com/v1/leaderboard"
PROFILE_URL = "https://gamma-api.polymarket.com/public-profile"

# https://docs.polymarket.com/api-reference/core/get-trader-leaderboard-rankings
CATEGORIES = (
    "OVERALL",
    "POLITICS",
    "SPORTS",
    "ESPORTS",
    "CRYPTO",
    "CULTURE",
    "MENTIONS",
    "WEATHER",
    "ECONOMICS",
    "TECH",
    "FINANCE",
)
TIME_PERIODS = ("DAY", "WEEK", "MONTH", "ALL")
ORDERINGS = ("PNL", "VOL")

MAX_PAGE_SIZE = 50
MAX_OFFSET = 1000

