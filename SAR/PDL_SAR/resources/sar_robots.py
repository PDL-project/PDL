"""
SAR robot definitions for PDL pipeline.
All SAR agents have the same skill set.
"""

sar_robot = {
    "name": "sar_agent",
    "skills": [
        "GoToObject",
        "GetSupply",
        "UseSupply",
        "Explore",
        "Carry",
        "DropOff",
        "StoreSupply",
    ],
    "mass": 100,
}

# Keep length >= max agents (10 for SAR)
robots = [sar_robot for _ in range(10)]
