import httpx
from langchain_core.tools import tool

_HTTP_TIMEOUT = 10.0
_FX_BASE_URL = "https://api.frankfurter.dev/v1"


def _fx_client() -> httpx.Client:
    return httpx.Client(timeout=_HTTP_TIMEOUT, follow_redirects=True)


def _geocode(client: httpx.Client, city: str) -> tuple[float, float, str] | None:
    geo = client.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
    ).json()
    results = geo.get("results") or []
    if not results:
        return None
    loc = results[0]
    name = f"{loc['name']}, {loc.get('country', '')}".strip(", ")
    return loc["latitude"], loc["longitude"], name


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city by name.

    Returns temperature in Celsius, wind speed, and a weather code.
    Uses the free Open-Meteo API.
    """
    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        geo = _geocode(client, city)
        if not geo:
            return f"Could not find a city named '{city}'."
        lat, lon, name = geo
        resp = client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,wind_speed_10m,weather_code",
            },
        ).json()

    current = resp["current"]
    return (
        f"Weather in {name}: {current['temperature_2m']}°C, "
        f"wind {current['wind_speed_10m']} km/h "
        f"(weather code {current['weather_code']})."
    )


@tool
def get_forecast(city: str, days: int = 3) -> str:
    """Get a multi-day daily weather forecast for a city.

    Args:
        city: City name (e.g. "Paris").
        days: Number of forecast days, 1-16. Defaults to 3.
    """
    if not 1 <= days <= 16:
        return "days must be between 1 and 16."

    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        geo = _geocode(client, city)
        if not geo:
            return f"Could not find a city named '{city}'."
        lat, lon, name = geo
        resp = client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
                "forecast_days": days,
                "timezone": "auto",
            },
        ).json()

    daily = resp["daily"]
    lines = [f"{days}-day forecast for {name}:"]
    for i, date in enumerate(daily["time"]):
        lines.append(
            f"  {date}: {daily['temperature_2m_min'][i]}°C to "
            f"{daily['temperature_2m_max'][i]}°C, "
            f"precip prob {daily['precipitation_probability_max'][i]}%, "
            f"weather code {daily['weather_code'][i]}"
        )
    return "\n".join(lines)


@tool
def get_air_quality(city: str) -> str:
    """Get the current air quality (US AQI, PM2.5, PM10) for a city."""
    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        geo = _geocode(client, city)
        if not geo:
            return f"Could not find a city named '{city}'."
        lat, lon, name = geo
        resp = client.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "us_aqi,pm2_5,pm10",
            },
        ).json()

    current = resp.get("current") or {}
    aqi = current.get("us_aqi")
    if aqi is None:
        return f"Air quality data is unavailable for {name}."
    return (
        f"Air quality in {name}: US AQI {aqi}, "
        f"PM2.5 {current.get('pm2_5')} µg/m³, "
        f"PM10 {current.get('pm10')} µg/m³."
    )


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount of money from one currency to another at the latest rate.

    Args:
        amount: Numeric amount to convert.
        from_currency: 3-letter ISO code of the source currency (e.g. USD, EUR, JPY).
        to_currency: 3-letter ISO code of the target currency (e.g. USD, EUR, JPY).
    """
    src = from_currency.upper()
    dst = to_currency.upper()

    with _fx_client() as client:
        resp = client.get(
            f"{_FX_BASE_URL}/latest",
            params={"amount": amount, "base": src, "symbols": dst},
        )

    if resp.status_code != 200:
        return f"Could not convert {src} to {dst}: {resp.text}"
    data = resp.json()
    value = data.get("rates", {}).get(dst)
    if value is None:
        return f"Could not convert {src} to {dst} (one of the codes may be unsupported)."
    return f"{amount:.2f} {src} = {value:.2f} {dst} (as of {data['date']})."


@tool
def get_historical_rate(from_currency: str, to_currency: str, date: str) -> str:
    """Get the exchange rate between two currencies on a specific past date.

    Args:
        from_currency: 3-letter ISO code (e.g. USD).
        to_currency: 3-letter ISO code (e.g. EUR).
        date: Date in YYYY-MM-DD format (e.g. "2024-01-15"). Must be in the past.
    """
    src = from_currency.upper()
    dst = to_currency.upper()

    with _fx_client() as client:
        resp = client.get(
            f"{_FX_BASE_URL}/{date}",
            params={"base": src, "symbols": dst},
        )

    if resp.status_code != 200:
        return f"Could not fetch historical rate for {date}: {resp.text}"
    data = resp.json()
    rate = data.get("rates", {}).get(dst)
    if rate is None:
        return f"Could not find rate {src} -> {dst} on {date}."
    return f"On {data['date']}: 1 {src} = {rate:.4f} {dst}."


@tool
def compare_currencies(
    amount: float, from_currency: str, to_currencies: list[str]
) -> str:
    """Convert one amount to multiple target currencies at once.

    Use this when the user is comparing values across several currencies in one
    question (more efficient than calling convert_currency repeatedly).

    Args:
        amount: Numeric amount to convert.
        from_currency: 3-letter ISO source code (e.g. USD).
        to_currencies: List of 3-letter target codes (e.g. ["EUR", "GBP", "JPY"]).
    """
    src = from_currency.upper()
    targets = ",".join(c.upper() for c in to_currencies)

    with _fx_client() as client:
        resp = client.get(
            f"{_FX_BASE_URL}/latest",
            params={"amount": amount, "base": src, "symbols": targets},
        )

    if resp.status_code != 200:
        return f"Could not compare currencies: {resp.text}"
    data = resp.json()
    rates = data.get("rates") or {}
    if not rates:
        return f"Could not convert {src} to {to_currencies}."

    lines = [f"{amount:.2f} {src} is:"]
    for code, value in rates.items():
        lines.append(f"  {value:.2f} {code}")
    return "\n".join(lines)
