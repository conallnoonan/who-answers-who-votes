"""State FIPS codes, postal abbreviations, and a tile-grid layout for maps."""
import pandas as pd

_STATES = [
    # fips, abbr, name, tile_row, tile_col
    (1, "AL", "Alabama", 6, 7), (2, "AK", "Alaska", 0, 0),
    (4, "AZ", "Arizona", 5, 2), (5, "AR", "Arkansas", 5, 5),
    (6, "CA", "California", 4, 1), (8, "CO", "Colorado", 4, 3),
    (9, "CT", "Connecticut", 3, 10), (10, "DE", "Delaware", 4, 10),
    (11, "DC", "District of Columbia", 5, 9), (12, "FL", "Florida", 7, 9),
    (13, "GA", "Georgia", 6, 8), (15, "HI", "Hawaii", 7, 0),
    (16, "ID", "Idaho", 2, 2), (17, "IL", "Illinois", 2, 6),
    (18, "IN", "Indiana", 3, 6), (19, "IA", "Iowa", 3, 5),
    (20, "KS", "Kansas", 5, 4), (21, "KY", "Kentucky", 4, 6),
    (22, "LA", "Louisiana", 6, 5), (23, "ME", "Maine", 0, 11),
    (24, "MD", "Maryland", 4, 9), (25, "MA", "Massachusetts", 2, 10),
    (26, "MI", "Michigan", 2, 7), (27, "MN", "Minnesota", 2, 5),
    (28, "MS", "Mississippi", 6, 6), (29, "MO", "Missouri", 4, 5),
    (30, "MT", "Montana", 2, 3), (31, "NE", "Nebraska", 4, 4),
    (32, "NV", "Nevada", 3, 2), (33, "NH", "New Hampshire", 1, 11),
    (34, "NJ", "New Jersey", 3, 9), (35, "NM", "New Mexico", 5, 3),
    (36, "NY", "New York", 2, 9), (37, "NC", "North Carolina", 5, 7),
    (38, "ND", "North Dakota", 2, 4), (39, "OH", "Ohio", 3, 7),
    (40, "OK", "Oklahoma", 6, 4), (41, "OR", "Oregon", 3, 1),
    (42, "PA", "Pennsylvania", 3, 8), (44, "RI", "Rhode Island", 3, 11),
    (45, "SC", "South Carolina", 5, 8), (46, "SD", "South Dakota", 3, 4),
    (47, "TN", "Tennessee", 5, 6), (48, "TX", "Texas", 7, 4),
    (49, "UT", "Utah", 4, 2), (50, "VT", "Vermont", 1, 10),
    (51, "VA", "Virginia", 4, 8), (53, "WA", "Washington", 2, 1),
    (54, "WV", "West Virginia", 4, 7), (55, "WI", "Wisconsin", 1, 6),
    (56, "WY", "Wyoming", 3, 3),
]

STATES = pd.DataFrame(_STATES, columns=["state_fips", "abbr", "name", "row", "col"])
ABBR = dict(zip(STATES.state_fips, STATES.abbr))
