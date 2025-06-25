"""FROM REFPYDST"""
from typing import Dict, Union, TypedDict, Optional, Literal

_TurnValue = Union[str, int, float]
Turn = Dict[str, _TurnValue]

# SlotNames are complete names including a domain and a slot, separated by a dash. e.g. "hotel-area"
SlotName = Literal["attraction-area", "attraction-name", "attraction-type", "bus-day", "bus-departure",
                   "bus-destination", "bus-leaveat", "hospital-department", "hotel-area", "hotel-book day",
                   "hotel-book people", "hotel-book stay", "hotel-internet", "hotel-name", "hotel-parking",
                   "hotel-pricerange", "hotel-stars", "hotel-type", "restaurant-area", "restaurant-book day",
                   "restaurant-book people", "restaurant-book time", "restaurant-food", "restaurant-name",
                   "restaurant-pricerange", "taxi-arriveby", "taxi-departure", "taxi-destination", "taxi-leaveat",
                   "train-arriveby", "train-book people", "train-day", "train-departure", "train-destination",
                   "train-leaveat"]

SlotValue = Union[str, int]  # Most commonly strings, but occasionally integers

# MultiWOZ Dict is the dictionary format for slot values as provided in the dataset. It is flattened, and denotes a
# dictionary that can be immediately evaluated using exact-match based metrics on keys and values. keys are in
# domain-slot form e.g. {"hotel-area": "centre", ...}
MultiWOZDict = Dict[SlotName, SlotValue]

