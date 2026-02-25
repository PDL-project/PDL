"""
SAR action definitions for PDL pipeline.
These strings describe the action signature that the LLM should use
when generating PDDL plans and task decompositions.
"""

sar_actions = [
    "GoToObject <robot> <object>",
    "GetSupply <robot> <reservoir_or_deposit> <supply_type>",
    "UseSupply <robot> <fire_region> <supply_type>",
    "Explore <robot>",
    "Carry <robot> <person>",
    "DropOff <robot> <person> <deposit>",
    "StoreSupply <robot> <deposit> <supply_type>",
]
sar_actions_str = ", ".join(sar_actions)
