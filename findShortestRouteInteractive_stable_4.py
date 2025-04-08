import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from geopy.distance import geodesic
from datetime import datetime, timedelta, time
import math
# Add this at the top with other imports
from streamlit.components.v1 import html

def parse_blocked_times(blocked_times_str):
    if not blocked_times_str:
        return []
    blocked_ranges = []
    ranges = [r.strip() for r in blocked_times_str.split(',')]
    for time_range in ranges:
        date_time_parts = time_range.split()
        date_str = date_time_parts[0]
        start_time = date_time_parts[1]
        end_time = date_time_parts[3]  # Skip the "-" at index 2
        
        # Parse date
        date_obj = datetime.strptime(date_str, '%d/%m/%Y')
        
        # Parse start and end times
        start_h, start_m = map(int, start_time.split(':'))
        end_h, end_m = map(int, end_time.split(':'))
        
        # Create full datetime objects
        start_datetime = date_obj.replace(hour=start_h, minute=start_m)
        end_datetime = date_obj.replace(hour=end_h, minute=end_m)
        
        blocked_ranges.append((start_datetime, end_datetime))
    return blocked_ranges

def is_time_blocked(current_time, duration, blocked_slots):
    slot_end = current_time + duration
    for blocked_start, blocked_end in blocked_slots:
        # Compare dates and times
        if (current_time.date() == blocked_start.date() and 
            not (slot_end <= blocked_start or current_time >= blocked_end)):
            return True
    return False


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

    return route


def calculate_travel_time_km(coord1, coord2):
    # Assume this function returns travel time in minutes based on coordinates
    # Placeholder for actual implementation
    return geodesic(coord1, coord2).km * 10  # Example: 10 minutes per km

def parse_holidays(holidays_str):
    if not holidays_str:
        return set()
    dates = [d.strip() for d in holidays_str.split(',')]
    return set(datetime.strptime(d, '%d/%m/%Y').date() for d in dates)

def add_working_days(start_date, days, holidays_set):
    """Add a given number of working days to a date, skipping weekends and holidays."""
    # Move start_date to the next working day if it falls on a weekend or holiday
    while start_date.weekday() >= 5 or start_date in holidays_set:
        start_date += timedelta(days=1)
    
    current = start_date
    while days > 0:
        current += timedelta(days=1)
        if current.weekday() < 5 and current not in holidays_set:
            days -= 1
    return current

def assign_timeslots_stable_with_travel_time(route_with_postal, inspection_times, blocked_slots=None, holidays_set=None, current_date=datetime.now().date(), start_hour=9, end_hour=18, max_distance_km=1, max_cases_per_slot=3):
    st.session_state.logs.append(f"Assigning timeslots with travel time for route: {route_with_postal}")
    st.session_state.logs.append(f"Inspection times: {inspection_times}")
    st.session_state.logs.append(f"Blocked slots: {blocked_slots}")
    st.session_state.logs.append(f"Start hour: {start_hour}, End hour: {end_hour}, Max distance: {max_distance_km} km, Max cases per slot: {max_cases_per_slot}")
    if blocked_slots is None:
        blocked_slots = []
        
    timeslots = []
    start_date = add_working_days(current_date, 2, holidays_set)
    base_time = datetime.combine(start_date, time(start_hour, 0))
    current_time = base_time
    group = []
    group_time = timedelta(hours=1)
    prev_group_end_time = base_time
    prev_group_last_coord = route_with_postal[0][0] if route_with_postal else None

    def move_to_next_day(current_time):
        st.session_state.logs.append(f"Moving to next day from {current_time}")
        next_day = current_time + timedelta(days=1)
        while next_day.weekday() >= 5 or next_day.date() in holidays_set:
            next_day += timedelta(days=1)
        return next_day.replace(hour=start_hour, minute=0, second=0, microsecond=0)

    def exceeds_end_hour(start_time, duration):
        end_time = start_time + duration
        result = end_time.hour > end_hour or (end_time.hour == end_hour and end_time.minute > 0)
        st.session_state.logs.append(f"Checking if exceeds end hour: Start {start_time}, Duration {duration}, Result {result}")
        return result

    def find_next_available_time(current_time, duration, blocked_slots, end_hour):
        while is_time_blocked(current_time, duration, blocked_slots) or exceeds_end_hour(current_time, duration):
            st.session_state.logs.append(f"Time blocked or exceeds end hour: Current {current_time}, Duration {duration}")
            current_time += timedelta(minutes=30)
            if current_time.hour >= end_hour or exceeds_end_hour(current_time, duration):
                current_time = move_to_next_day(current_time)
        st.session_state.logs.append(f"Next available time: {current_time}")
        return current_time

    for i, (coord, postal) in enumerate(route_with_postal):
        inspection_time = inspection_times[i]
        case = (coord, postal, inspection_time)

        if inspection_time > 1:
            if group:
                current_time = find_next_available_time(current_time, group_time, blocked_slots, end_hour)
                if exceeds_end_hour(current_time, group_time):
                    current_time = move_to_next_day(current_time)
                group_end = current_time + group_time
                st.session_state.logs.append(f"Assigning group timeslot: Start {current_time}, End {group_end}")
                timeslots.append((group, current_time, group_end.strftime("%H:%M")))
                prev_group_end_time = group_end
                prev_group_last_coord = group[-1][0] if group else prev_group_last_coord
                current_time = group_end
                group = []

            if prev_group_end_time is not None:
                travel_time_min = calculate_travel_time_km(prev_group_last_coord, coord)
                rounded_travel_time = timedelta(minutes=math.ceil(travel_time_min / 30) * 30)
                new_start_time = prev_group_end_time + rounded_travel_time
                current_time = find_next_available_time(new_start_time, timedelta(hours=inspection_time), blocked_slots, end_hour)
                if exceeds_end_hour(current_time, timedelta()):
                    current_time = move_to_next_day(current_time)

            duration = timedelta(hours=inspection_time)
            if exceeds_end_hour(current_time, duration):
                current_time = move_to_next_day(current_time)
            
            current_time = find_next_available_time(current_time, duration, blocked_slots, end_hour)
            start = current_time
            end = current_time + duration
            st.session_state.logs.append(f"Assigning single case timeslot: Start {start}, End {end}")
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
                    current_time = find_next_available_time(new_start_time, group_time, blocked_slots, end_hour)
                    if exceeds_end_hour(current_time, timedelta()):
                        current_time = move_to_next_day(current_time)
                
                current_time = find_next_available_time(current_time, group_time, blocked_slots, end_hour)
                if exceeds_end_hour(current_time, group_time):
                    current_time = move_to_next_day(current_time)
                group.append(case)
            else:
                last_coord, _, _ = group[-1]
                distance_km = geodesic(last_coord, coord).km
                st.session_state.logs.append(f"Distance from {last_coord} to {coord}: {distance_km} km")
                if distance_km < max_distance_km and len(group) < max_cases_per_slot:
                    group.append(case)
                else:
                    current_time = find_next_available_time(current_time, group_time, blocked_slots, end_hour)
                    if exceeds_end_hour(current_time, group_time):
                        current_time = move_to_next_day(current_time)
                    group_end = current_time + group_time
                    st.session_state.logs.append(f"Assigning group timeslot: Start {current_time}, End {group_end}")
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
        current_time = find_next_available_time(current_time, group_time, blocked_slots, end_hour)
        if exceeds_end_hour(current_time, group_time):
            current_time = move_to_next_day(current_time)
        group_end = current_time + group_time
        st.session_state.logs.append(f"Assigning final group timeslot: Start {current_time}, End {group_end}")
        timeslots.append((group, current_time, group_end.strftime("%H:%M")))

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
if "blocked_slots" not in st.session_state:
    st.session_state.blocked_slots = []
if "blocked_slots" not in st.session_state:
    st.session_state.blocked_slots = []
if "current_date" not in st.session_state:
    st.session_state.current_date = []
if "logs" not in st.session_state:
    st.session_state.logs = []


# Add this function to create a popup-like experience
def show_logs():
    with st.expander("🐛 Debug Logs", expanded=False):
        if st.button("Clear Logs"):
            st.session_state.logs = []
        log_text = "\n".join(st.session_state.logs)
        st.code(log_text, language="plaintext")

# Streamlit App
st.set_page_config(layout="centered")
st.title("📍 Route Optimizer with Time Slots")

# Add blocked time slots input
st.sidebar.header("Additional Settings")

blocked_times = st.sidebar.text_input(
    "Enter blocked time slots (format: '26/04/2025 9:30 - 10:30, 30/04/2025 13:00 - 16:30')",
    "07/04/2025 9:30 - 10:30, 08/04/2025 8:00 - 18:00, 30/04/2025 13:00 - 16:30",  # Default value
    help="Input Lunch Breaks, Leaves, Holiday, Appoinment. If whole day is blocked, use 'dd/mm/yyyy 9:00 - 18:00'."
    " Use comma to separate multiple time slots. Example: '26/04/2025 9:30 - 10:30, 30/04/2025 13:00 - 16:30'"
    " (24-hour format)."
)
# Add the new holiday input
holidays = st.sidebar.multiselect(
    "Holidays",
    options=[(datetime.now() + timedelta(days=i)).strftime("%d/%m/%Y") for i in range(0, 365)],
    default=["08/04/2025", "09/04/2025"],
    help="Select dates where the whole day is blocked (e.g., public holidays)."
)
current_date = st.sidebar.date_input("Current Date", datetime.now().date(), format="DD/MM/YYYY")
st.session_state.current_date = current_date
st.sidebar.write("Blocked Time Slots:", blocked_times)
holidays_str = ", ".join(holidays)
st.sidebar.write("Holidays:", holidays_str)
st.sidebar.write("Current Date:", current_date.strftime("%d/%m/%Y"))


if blocked_times:
    st.session_state.blocked_slots = parse_blocked_times(blocked_times)
if holidays:
    st.session_state.holiday_set = parse_holidays(holidays_str)


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


show_logs()

# Button actions
col1, col2 = st.columns(2)
with col1:
    if st.button("🧮 Recalculate Shortest Route"):
        if len(st.session_state.points) < 2:
            st.warning("Add at least 2 points.")
        else:
            startCoords = st.session_state.points[0]
            shortest_route = find_shortest_path(st.session_state.points, startCoords)
            route_with_postal = [(coord, f"P{i+1:03}") for i, coord in enumerate(shortest_route)]
            
            # Get inspection time inputs (ensure they retain the previous values)
            inspection_times = []
            for i in range(len(st.session_state.points)):
                inspection_time = st.session_state.inspection_times.get(i, 1)  # Default to 1 if not set
                inspection_times.append(inspection_time)
            
            # When calling assign_timeslots_stable_with_travel_time, add blocked_slots parameter:
            route_with_timeslot = assign_timeslots_stable_with_travel_time(
                route_with_postal=route_with_postal,
                inspection_times=inspection_times,
                blocked_slots=st.session_state.blocked_slots,
                holidays_set=st.session_state.holiday_set,
                current_date=st.session_state.current_date
            )
            st.session_state.route_data = route_with_timeslot
            st.session_state.show_route = True
            # Redraw the map with the new route
            st.rerun()


with col2:
    if st.button("🔄 Reset Points"):
        st.session_state.points = []
        st.session_state.show_route = False
        st.session_state.route_data = None
        st.session_state.inspection_times = {}
        st.rerun()

# Display current points (with correct order based on the route)
if st.session_state.points:
    st.session_state.logs.append("Current points:")
    for i, point in enumerate(st.session_state.points):
        lat, lng = point
        # Generate the label based on the current index
        label = f"P{i+1:03}"
        st.session_state.logs.append(f"Point {label}: Latitude {lat:.4f}, Longitude {lng:.4f}")
        
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
            #break  # Break to avoid mutating the list while iterating over it
