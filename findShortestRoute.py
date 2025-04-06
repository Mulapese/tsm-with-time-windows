import requests
import folium
from geopy.distance import geodesic
from itertools import permutations
from datetime import datetime, timedelta

ONEMAP_API = "https://www.onemap.gov.sg/api/common/elastic/search"

def get_coordinates(postal_code):
    params = {
        "searchVal": postal_code,
        "returnGeom": "Y",
        "getAddrDetails": "Y",
        "pageNum": 1
    }
    response = requests.get(ONEMAP_API, params=params)
    data = response.json()
    results = data.get("results", [])
    if not results:
        raise ValueError(f"No result for postal code: {postal_code}")
    lat = float(results[0]["LATITUDE"])
    lon = float(results[0]["LONGITUDE"])
    return (lat, lon)

def total_distance(route):
    return sum(geodesic(route[i], route[i+1]).km for i in range(len(route) - 1))

def find_shortest_path(coords, startCoords):
    remaining = [c for c in coords if c != startCoords]
    min_route = None
    min_dist = float("inf")
    for perm in permutations(remaining):
        route = [startCoords] + list(perm)
        dist = total_distance(route)
        if dist < min_dist:
            min_dist = dist
            min_route = route
    return min_route, min_dist

def assign_timeslots(route_with_postal, start_hour=9):
    timeslots = []
    base_time = datetime.strptime(f"{start_hour}:00", "%H:%M")
    for i, (coord, postal) in enumerate(route_with_postal):
        start = (base_time + timedelta(hours=i)).strftime("%H:%M")
        end = (base_time + timedelta(hours=i + 1)).strftime("%H:%M")
        timeslots.append((coord, postal, f"{start} - {end}"))
    return timeslots

def plot_route(route_with_postal_timeslot, output_file="shortest_route_map.html"):
    m = folium.Map(location=route_with_postal_timeslot[0][0], zoom_start=13)

    for i, (coord, postal, timeslot) in enumerate(route_with_postal_timeslot):
        color = "green" if i == 0 else ("red" if i == len(route_with_postal_timeslot) - 1 else "blue")
        
        # Pin marker
        folium.Marker(coord, icon=folium.Icon(color=color)).add_to(m)

        # Always visible label
        folium.map.Marker(
            location=coord,
            icon=folium.DivIcon(
                icon_size=(150, 36),
                html=f'<div style="font-size: 12px; color: black; background-color: white; padding: 2px; border-radius: 4px; border: 1px solid grey;">{postal}<br>{timeslot}</div>'
            )
        ).add_to(m)

    # Draw lines and distances
    for i in range(len(route_with_postal_timeslot) - 1):
        c1, _, _ = route_with_postal_timeslot[i]
        c2, _, _ = route_with_postal_timeslot[i + 1]
        dist_km = geodesic(c1, c2).km
        midpoint = ((c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2)
        folium.PolyLine([c1, c2], color="blue", weight=3).add_to(m)
        folium.Marker(midpoint, icon=folium.DivIcon(
            html=f'<div style="font-size: 11px; color: black;">{dist_km:.2f} km</div>'
        )).add_to(m)

    m.save(output_file)
    print(f"Map saved to {output_file}")


def read_postal_list(filename="postalList.txt"):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip()]

def find_and_plot(postal_codes):
    coords_with_postal = [(get_coordinates(p), p) for p in postal_codes]
    startCoords = coords_with_postal[0][0]
    route_coords = [x[0] for x in coords_with_postal]
    shortest_route, _ = find_shortest_path(route_coords, startCoords)

    # Match route back to postal codes
    coord_to_postal = {coord: postal for coord, postal in coords_with_postal}
    route_with_postal = [(coord, coord_to_postal[coord]) for coord in shortest_route]

    # Assign time slots
    route_with_postal_timeslot = assign_timeslots(route_with_postal)
    
    # Show map
    plot_route(route_with_postal_timeslot)

# Main
if __name__ == "__main__":
    postal_list = read_postal_list("postalList.txt")
    find_and_plot(postal_list)
