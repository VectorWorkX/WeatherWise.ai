#!/usr/bin/env python3
"""Test script for Xweather MCP tools.

Run with: python test_tools.py
"""

import asyncio
import json
import os
from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    # Get credentials from environment or use defaults
    client_id = os.getenv("XWEATHER_CLIENT_ID", "hzTzHMjpIf4p1a08Os4s1")
    client_secret = os.getenv(
        "XWEATHER_CLIENT_SECRET", "fYGCNFHHa9KPlHIrg6At2mygmJtVnPReEvBlBhe7"
    )
    api_key = f"{client_id}_{client_secret}"

    client = MultiServerMCPClient(
        {
            "xweather": {
                "url": "https://mcp.api.xweather.com/mcp",
                "transport": "streamable_http",
                "headers": {"Authorization": f"Bearer {api_key}"},
            }
        }
    )

    print("🌍 Connecting to Xweather MCP Server...")
    tools = await client.get_tools()
    tools_dict = {t.name: t for t in tools}
    print(f"✅ Connected! Found {len(tools)} tools\n")

    # Test 1: Current Weather
    print("=" * 50)
    print("🌤️  TEST 1: Current Weather - San Francisco")
    print("=" * 50)
    tool = tools_dict["xweather_get_current_weather"]
    result = await tool.ainvoke({"location": "San Francisco, CA"})
    data = json.loads(result[0]["text"])
    w = data["weather"]
    print(f"Location: {data['place']['city']}, {data['place']['state']}")
    print(f"Weather:  {w['weather']}")
    print(f"Temp:     {w['tempF']}°F ({w['tempC']}°C)")
    print(f"Humidity: {w['humidity']}%")
    print(f"Wind:     {w['windSpeedMPH']} MPH {w['windDir']}")
    if data.get("alerts"):
        print(f"⚠️ Alert:  {data['alerts'][0]['title']}")
    print()

    # Test 2: Air Quality
    print("=" * 50)
    print("🌬️  TEST 2: Air Quality - Los Angeles")
    print("=" * 50)
    tool = tools_dict["xweather_get_current_airquality"]
    result = await tool.ainvoke({"location": "Los Angeles, CA"})
    data = json.loads(result[0]["text"])
    aq = data["air_quality"]
    print(f"Location: {data['place']['city']}, {data['place']['state']}")
    print(f"AQI:      {aq['aqi']} ({aq['category']})")
    print()

    # Test 3: Weather Impacts
    print("=" * 50)
    print("🏃 TEST 3: Weather Impacts - Seattle")
    print("=" * 50)
    tool = tools_dict["xweather_get_current_weather_impacts"]
    result = await tool.ainvoke({"location": "Seattle, WA"})
    data = json.loads(result[0]["text"])
    print(f"Location: {data['place']['city']}, {data['place']['state']}")
    if "impacts" in data:
        for impact in data["impacts"][:3]:
            print(f"  • {impact.get('activity', 'N/A')}: {impact.get('impact', 'N/A')}")
    print()

    # Test 4: Precipitation Timing
    print("=" * 50)
    print("🌧️  TEST 4: Precipitation Timing - Chicago")
    print("=" * 50)
    tool = tools_dict["xweather_get_forecast_precipitation_timing"]
    result = await tool.ainvoke({"location": "Chicago, IL"})
    data = json.loads(result[0]["text"])
    print(f"Location: {data['place']['city']}, {data['place']['state']}")
    timing = data.get("precipitation_timing", {})
    print(f"Summary:  {timing.get('summary', 'No precipitation expected')}")
    print()

    # Test 5: Active Tropical Systems
    print("=" * 50)
    print("🌀 TEST 5: Active Tropical Systems")
    print("=" * 50)
    tool = tools_dict["xweather_get_current_active_tropical_systems"]
    result = await tool.ainvoke({})
    data = json.loads(result[0]["text"])
    systems = data.get("tropical_systems", [])
    if systems:
        for system in systems[:3]:
            print(f"  • {system.get('name', 'Unknown')}: {system.get('type', 'N/A')}")
    else:
        print("  No active tropical systems")
    print()

    print("=" * 50)
    print("✅ All tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
