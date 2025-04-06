import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from geopy.distance import geodesic
from datetime import datetime, timedelta
import math


# Utils
def total_distance(route):
    return sum(geodesic(route[i], route[i+1]).km for i in range(len(route) - 1))

def find_shortest_path(coords, startCoords):
    # Start with the starting coordinate
    current_coord = startCoords
    route = [current_coord]
    remaining = set(coords) - {startCoords}  # Coordinates left to visit
    
    # Keep finding the nearest neighbor
    while remaining:
        nearest_coord = min(remaining, key=lambda c: geodesic(current_coord, c).km)
        route.append(nearest_coord)
        remaining.remove(nearest_coord)
        current_coord = nearest_coord

    # Calculate total distance
    dist = total_distance(route)
    return route, dist


# Fix to ensure inspection_time is included in the group
def assign_timeslots_stable_without_travel_time(route_with_postal, inspection_times, start_hour=9, end_hour=18, max_distance_km=1, max_cases_per_slot=3):
    timeslots = []
    base_time = datetime.now().replace(hour=start_hour, minute=0, second=0, microsecond=0)
    current_time = base_time
    group = []
    group_time = timedelta(hours=1)

    def move_to_next_day(current_time):
        return (current_time + timedelta(days=1)).replace(hour=start_hour, minute=0, second=0, microsecond=0)

    def exceeds_end_hour(start_time, duration):
        end_time = start_time + duration
        return end_time.hour > end_hour or (end_time.hour == end_hour and end_time.minute > 0)

    for i, (coord, postal) in enumerate(route_with_postal):
        inspection_time = inspection_times[i]
        case = (coord, postal, inspection_time)

        if inspection_time > 1:
            duration = timedelta(hours=inspection_time)
            if exceeds_end_hour(current_time, duration):
                current_time = move_to_next_day(current_time)
            if group:
                if exceeds_end_hour(current_time, group_time):
                    current_time = move_to_next_day(current_time)
                timeslots.append((group, current_time, (current_time + group_time).strftime("%H:%M")))
                current_time += group_time
                group = []
            timeslots.append(([case], current_time, (current_time + duration).strftime("%H:%M")))
            current_time += duration
        else:
            if not group:
                group.append(case)
            else:
                last_coord, _, _ = group[-1]
                if geodesic(last_coord, coord).km < max_distance_km and len(group) < max_cases_per_slot:
                    group.append(case)
                else:
                    if exceeds_end_hour(current_time, group_time):
                        current_time = move_to_next_day(current_time)
                    timeslots.append((group, current_time, (current_time + group_time).strftime("%H:%M")))
                    current_time += group_time
                    group = [case]

    if group:
        if exceeds_end_hour(current_time, group_time):
            current_time = move_to_next_day(current_time)
        timeslots.append((group, current_time, (current_time + group_time).strftime("%H:%M")))

    result_timeslots = []
    for group, start_time, end_time in timeslots:
        time_slot = f"({start_time.strftime('%d/%m, %H:%M')} - {end_time})"
        for coord, postal, _ in group:
            result_timeslots.append((coord, postal, time_slot))

    return result_timeslots


def calculate_travel_time_km(coord1, coord2):
    # Assume this function returns travel time in minutes based on coordinates
    # Placeholder for actual implementation
    return geodesic(coord1, coord2).km * 10  # Example: 2 minutes per km

def assign_timeslots_stable_with_travel_time(route_with_postal, inspection_times, start_hour=9, end_hour=18, max_distance_km=1, max_cases_per_slot=3):
    timeslots = []
    base_time = datetime.now().replace(hour=start_hour, minute=0, second=0, microsecond=0)
    current_time = base_time
    group = []
    group_time = timedelta(hours=1)
    prev_group_end_time = None
    prev_group_last_coord = None

    def move_to_next_day(current_time):
        return (current_time + timedelta(days=1)).replace(hour=start_hour, minute=0, second=0, microsecond=0)

    def exceeds_end_hour(start_time, duration):
        end_time = start_time + duration
        return end_time.hour > end_hour or (end_time.hour == end_hour and end_time.minute > 0)

    for i, (coord, postal) in enumerate(route_with_postal):
        inspection_time = inspection_times[i]
        case = (coord, postal, inspection_time)

        if inspection_time > 1:
            if group:
                if exceeds_end_hour(current_time, group_time):
                    current_time = move_to_next_day(current_time)
                group_end = current_time + group_time
                timeslots.append((group, current_time, group_end.strftime("%H:%M")))
                prev_group_end_time = group_end
                prev_group_last_coord = group[-1][0]
                current_time = group_end
                group = []

            if prev_group_end_time is not None:
                travel_time_min = calculate_travel_time_km(prev_group_last_coord, coord)
                rounded_travel_time = timedelta(minutes=math.ceil(travel_time_min / 30) * 30)
                new_start_time = prev_group_end_time + rounded_travel_time
                if exceeds_end_hour(new_start_time, timedelta()):
                    new_start_time = move_to_next_day(new_start_time)
                current_time = new_start_time

            duration = timedelta(hours=inspection_time)
            if exceeds_end_hour(current_time, duration):
                current_time = move_to_next_day(current_time)
            
            start = current_time
            end = current_time + duration
            timeslots.append(([case], start, end.strftime("%H:%M")))
            prev_group_end_time = end
            prev_group_last_coord = coord
            current_time = end
        else:
            if not group:
                if prev_group_end_time is not None:
                    travel_time_min = calculate_travel_time_km(prev_group_last_coord, coord)
                    rounded_travel_time = timedelta(minutes=math.ceil(travel_time_min / 30) * 30)
                    new_start_time = prev_group_end_time + rounded_travel_time
                    if exceeds_end_hour(new_start_time, timedelta()):
                        new_start_time = move_to_next_day(new_start_time)
                    current_time = new_start_time
                
                if exceeds_end_hour(current_time, group_time):
                    current_time = move_to_next_day(current_time)
                group.append(case)
            else:
                last_coord, _, _ = group[-1]
                distance_km = geodesic(last_coord, coord).km
                if distance_km < max_distance_km and len(group) < max_cases_per_slot:
                    group.append(case)
                else:
                    if exceeds_end_hour(current_time, group_time):
                        current_time = move_to_next_day(current_time)
                    group_end = current_time + group_time
                    timeslots.append((group, current_time, group_end.strftime("%H:%M")))
                    prev_group_end_time = group_end
                    prev_group_last_coord = group[-1][0]
                    current_time = group_end
                    group = [case]
                    
                    # Calculate travel time after finalizing previous group and starting new group
                    if prev_group_end_time is not None:
                        travel_time_min = calculate_travel_time_km(prev_group_last_coord, coord)
                        rounded_travel_time = timedelta(minutes=math.ceil(travel_time_min / 30) * 30)
                        new_start_time = prev_group_end_time + rounded_travel_time
                        if exceeds_end_hour(new_start_time, timedelta()):
                            new_start_time = move_to_next_day(new_start_time)
                        current_time = new_start_time
                    
                    if exceeds_end_hour(current_time, group_time):
                        current_time = move_to_next_day(current_time)

    if group:
        if exceeds_end_hour(current_time, group_time):
            current_time = move_to_next_day(current_time)
        group_end = current_time + group_time
        timeslots.append((group, current_time, group_end.strftime("%H:%M")))
        prev_group_end_time = group_end
        prev_group_last_coord = group[-1][0]

    result_timeslots = []
    for group, start_time, end_time_str in timeslots:
        start_time_str = start_time.strftime("%d/%m, %H:%M")
        time_slot = f"({start_time_str} - {end_time_str})"
        for coord, postal, _ in group:
            result_timeslots.append((coord, postal, time_slot))
    
    return result_timeslots


# Initialize session state
if "points" not in st.session_state:
    st.session_state.points = []
if "show_route" not in st.session_state:
    st.session_state.show_route = False
if "route_data" not in st.session_state:
    st.session_state.route_data = None
if "inspection_times" not in st.session_state:
    st.session_state.inspection_times = {}

# Streamlit App
st.set_page_config(layout="centered")
st.title("📍 Route Optimizer with Time Slots")

# Create the base map
if st.session_state.route_data:
    # Show the route map if we have calculated a route
    route_with_timeslot = st.session_state.route_data
    m = folium.Map(location=route_with_timeslot[0][0], zoom_start=13)
    
    # Draw the route
    for i, (coord, postal, timeslot) in enumerate(route_with_timeslot):
        color = "green" if i == 0 else ("red" if i == len(route_with_timeslot) - 1 else "blue")
        folium.Marker(coord, icon=folium.Icon(color=color)).add_to(m)
        folium.map.Marker(
            location=coord,
            icon=folium.DivIcon(
                icon_size=(150, 36),
                html=f'<div style="font-size: 12px; color: black; background-color: white; padding: 2px; border-radius: 4px; border: 1px solid grey;">{postal}<br>{timeslot}</div>'
            )
        ).add_to(m)

    for i in range(len(route_with_timeslot) - 1):
        c1, _, _ = route_with_timeslot[i]
        c2, _, _ = route_with_timeslot[i + 1]
        dist_km = geodesic(c1, c2).km
        midpoint = ((c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2)
        folium.PolyLine([c1, c2], color="blue", weight=3).add_to(m)
        folium.Marker(midpoint, icon=folium.DivIcon(
            html=f'<div style="font-size: 11px; color: black;">{dist_km:.2f} km</div>'
        )).add_to(m)
    
    # Add drawing tools to the route map
    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "rectangle": False,
            "circle": False,
            "circlemarker": False,
            "polygon": False
        },
        edit_options={"edit": False, "remove": True}
    ).add_to(m)
    
else:
    # Show the default drawing map if no route calculated yet
    m = folium.Map(location=[1.3521, 103.8198], zoom_start=12)
    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "rectangle": False,
            "circle": False,
            "circlemarker": False,
            "polygon": False
        },
        edit_options={"edit": False, "remove": True}
    ).add_to(m)

# Display the map and handle drawn points
output = st_folium(m, width=700, height=500, key="map", returned_objects=["last_active_drawing", "all_drawings"])

# Update drawn points
if output["last_active_drawing"]:
    coords = output["last_active_drawing"]["geometry"]["coordinates"]
    latlng = (coords[1], coords[0])
    if latlng not in st.session_state.points:
        st.session_state.points.append(latlng)
        st.session_state.show_route = False  # Reset route when new points are added

# Recalculate the route and reindex the points based on the route order
if st.session_state.show_route and st.session_state.route_data:
    route_with_timeslot = st.session_state.route_data
    # Reorder points based on the route order
    ordered_points = [coord for coord, _, _ in route_with_timeslot]
    st.session_state.points = ordered_points  # Update session state to reflect the correct order

# Button actions
col1, col2 = st.columns(2)
with col1:
    if st.button("🧮 Recalculate Shortest Route"):
        if len(st.session_state.points) < 2:
            st.warning("Add at least 2 points.")
        else:
            startCoords = st.session_state.points[0]
            shortest_route, dist = find_shortest_path(st.session_state.points, startCoords)
            route_with_postal = [(coord, f"P{i+1:03}") for i, coord in enumerate(shortest_route)]
            
            # Get inspection time inputs (ensure they retain the previous values)
            inspection_times = []
            for i in range(len(st.session_state.points)):
                inspection_time = st.session_state.inspection_times.get(i, 1)  # Default to 1 if not set
                inspection_times.append(inspection_time)
            
            route_with_timeslot = assign_timeslots_stable_with_travel_time(route_with_postal, inspection_times)
            st.session_state.route_data = route_with_timeslot
            st.session_state.show_route = True
            st.success(f"✅ Shortest distance: {dist:.2f} km")

with col2:
    if st.button("🔄 Reset Points"):
        st.session_state.points = []
        st.session_state.show_route = False
        st.session_state.route_data = None
        st.session_state.inspection_times = {}

# Display current points (with correct order based on the route)
if st.session_state.points:
    st.write("Current points:")
    for i, point in enumerate(st.session_state.points):
        lat, lng = point
        # Generate the label based on the current index
        label = f"P{i+1:03}"
        st.write(f"Point {label}: Latitude {lat:.4f}, Longitude {lng:.4f}")
        
        # Add an inspection time input for each point
        inspection_time = st.number_input(f"Inspection Time for Point {i+1}: {label}", min_value=1, max_value=8, value=st.session_state.inspection_times.get(i, 1), key=f"inspection_time_{i}")
        st.session_state.inspection_times[i] = inspection_time  # Store the input value
        
        # Add a delete button for each point
        delete_button = st.button(f"❌ Delete {label}", key=f"delete_{i}")
        if delete_button:
            st.session_state.points.pop(i)
            st.session_state.show_route = False  # Reset the route when a point is deleted
            st.session_state.route_data = None  # Clear the route data
            st.success(f"✅ Point {label} deleted successfully.")
            break  # Break to avoid mutating the list while iterating over it
