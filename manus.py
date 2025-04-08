import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from geopy.distance import geodesic
from datetime import datetime, timedelta, time
import math
# Add this at the top with other imports
from streamlit.components.v1 import html

#region Logic
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

def parse_resolved_cases(resolved_cases):
    """
    Parse resolved cases into a format suitable for timeslot assignment.
    
    Parameters:
    - resolved_cases: List of dictionaries with location, timeslot, and case_id.
    
    Returns:
    - List of tuples (start_datetime, end_datetime, location) for each resolved case.
    """
    parsed_cases = []
    for case in resolved_cases:
        location = case["location"]
        timeslot = case["timeslot"]
        
        # Parse the timeslot string format "(dd/mm, HH:MM - HH:MM)"
        timeslot = timeslot.strip("()")
        date_part, time_part = timeslot.split(", ")
        start_time_str, end_time_str = time_part.split(" - ")
        
        # Parse date (assuming current year)
        day, month = map(int, date_part.split("/"))
        current_year = datetime.now().year
        date_obj = datetime(current_year, month, day)
        
        # Parse start and end times
        start_h, start_m = map(int, start_time_str.split(":"))
        end_h, end_m = map(int, end_time_str.split(":"))
        
        # Create full datetime objects
        start_datetime = date_obj.replace(hour=start_h, minute=start_m)
        end_datetime = date_obj.replace(hour=end_h, minute=end_m)
        
        parsed_cases.append((start_datetime, end_datetime, location))
    
    # Sort by start time
    parsed_cases.sort(key=lambda x: x[0])
    return parsed_cases

def get_resolved_case_timeslots(parsed_resolved_cases):
    """
    Extract timeslots from parsed resolved cases.
    
    Parameters:
    - parsed_resolved_cases: List of tuples (start_datetime, end_datetime, location).
    
    Returns:
    - List of tuples (start_datetime, end_datetime) representing blocked timeslots.
    """
    return [(start, end) for start, end, _ in parsed_resolved_cases]

def get_resolved_case_locations_with_times(parsed_resolved_cases):
    """
    Extract locations with their associated times from parsed resolved cases.
    
    Parameters:
    - parsed_resolved_cases: List of tuples (start_datetime, end_datetime, location).
    
    Returns:
    - List of tuples (location, start_datetime, end_datetime) sorted by start time.
    """
    return [(loc, start, end) for start, end, loc in parsed_resolved_cases]

def assign_timeslots_with_resolved_cases(open_cases_with_postal, inspection_times, resolved_cases=[], blocked_slots=None, holidays_set=None, current_date=datetime.now().date(), start_hour=9, end_hour=18, max_distance_km=1, max_cases_per_slot=3):
    """
    Assign timeslots for inspection cases, considering both open cases and resolved cases.
    
    Parameters:
    - open_cases_with_postal: List of tuples (coordinate, postal code) representing open cases.
    - inspection_times: List of inspection duration (in hours) corresponding to each open case.
    - resolved_cases: List of dictionaries for resolved cases with location, timeslot, and case_id.
    - blocked_slots: List of blocked time intervals (tuples of start and end datetime).
    - holidays_set: Set of holiday dates.
    - current_date: The current date from which scheduling starts.
    - start_hour: Hour of the day when inspections can start.
    - end_hour: Hour of the day by which inspections must end.
    - max_distance_km: Maximum allowed distance between consecutive cases to group them.
    - max_cases_per_slot: Maximum number of cases allowed in a single timeslot group.
    
    Returns:
    - A list of tuples (coordinate, postal, timeslot_str) where timeslot_str is formatted as "(dd/mm, HH:MM - HH:MM)".
    """
    st.session_state.logs.append(f"Assigning timeslots with resolved cases for open cases: {open_cases_with_postal}")
    st.session_state.logs.append(f"Inspection times: {inspection_times}")
    st.session_state.logs.append(f"Resolved cases: {resolved_cases}")
    st.session_state.logs.append(f"Blocked slots: {blocked_slots}")
    st.session_state.logs.append(f"Start hour: {start_hour}, End hour: {end_hour}, Max distance: {max_distance_km} km, Max cases per slot: {max_cases_per_slot}")
    
    if blocked_slots is None:
        blocked_slots = []
    
    # If no open cases, return empty list
    if not open_cases_with_postal:
        return []
    
    # If no resolved cases, use the original algorithm
    if not resolved_cases:
        return assign_timeslots_stable_with_travel_time(
            route_with_postal=open_cases_with_postal,
            inspection_times=inspection_times,
            resolved_cases=[],
            blocked_slots=blocked_slots,
            holidays_set=holidays_set,
            current_date=current_date,
            start_hour=start_hour,
            end_hour=end_hour,
            max_distance_km=max_distance_km,
            max_cases_per_slot=max_cases_per_slot
        )
    
    # Parse resolved cases to get their timeslots and locations
    parsed_resolved_cases = parse_resolved_cases(resolved_cases)
    resolved_timeslots = get_resolved_case_timeslots(parsed_resolved_cases)
    resolved_locations_with_times = get_resolved_case_locations_with_times(parsed_resolved_cases)
    
    # Add resolved case timeslots to blocked slots
    all_blocked_slots = blocked_slots + resolved_timeslots
    
    # Sort resolved cases by start time
    resolved_locations_with_times.sort(key=lambda x: x[1])
    
    # Initialize result list for all cases (open and resolved)
    result_timeslots = []
    
    # Start with the first open case as the starting point if no resolved cases for today
    remaining_open_cases = [case for case, _ in open_cases_with_postal]
    remaining_open_postals = [postal for _, postal in open_cases_with_postal]
    remaining_inspection_times = inspection_times.copy()
    
    # Start date for scheduling
    start_date = add_working_days(current_date, 2, holidays_set)
    base_time = datetime.combine(start_date, time(start_hour, 0))
    current_time = base_time
    
    # Helper functions
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
    
    # Process each day with resolved cases
    resolved_days = {loc_time[1].date() for loc_time in resolved_locations_with_times}
    
    # First, handle days with resolved cases
    for day in sorted(resolved_days):
        day_resolved_cases = [loc_time for loc_time in resolved_locations_with_times if loc_time[1].date() == day]
        day_resolved_cases.sort(key=lambda x: x[1])  # Sort by start time
        
        # Skip if this day is before our start date
        if day < start_date:
            continue
        
        # Set current time to the start of this day
        current_time = datetime.combine(day, time(start_hour, 0))
        
        # For each resolved case on this day
        for resolved_idx, (resolved_loc, resolved_start, resolved_end) in enumerate(day_resolved_cases):
            # If we have open cases to process
            if remaining_open_cases:
                # Find the shortest path from the current resolved case
                if resolved_idx == 0:
                    # For the first resolved case of the day, find path from an open case to this resolved case
                    startCoord = remaining_open_cases[0]
                else:
                    # For subsequent resolved cases, start from the previous resolved case
                    startCoord = day_resolved_cases[resolved_idx-1][0]
                
                # Calculate shortest path for remaining open cases
                shortest_route = find_shortest_path(remaining_open_cases, startCoord)
                
                # Process open cases that can fit before this resolved case
                current_time = datetime.combine(day, time(start_hour, 0)) if resolved_idx == 0 else day_resolved_cases[resolved_idx-1][2]
                
                # Try to fit open cases before this resolved case
                i = 0
                while i < len(shortest_route) and remaining_open_cases:
                    coord = shortest_route[i]
                    if coord == startCoord and i > 0:
                        i += 1
                        continue
                    
                    # Find the index in the remaining open cases
                    try:
                        idx = remaining_open_cases.index(coord)
                        inspection_time = remaining_inspection_times[idx]
                        postal = remaining_open_postals[idx]
                        
                        # Calculate duration
                        duration = timedelta(hours=inspection_time)
                        
                        # Calculate travel time to next location
                        travel_time_to_next = calculate_travel_time_km(coord, resolved_loc)
                        travel_time_delta = timedelta(minutes=math.ceil(travel_time_to_next / 30) * 30)
                        
                        # Check if we can fit this case before the resolved case
                        potential_end_time = current_time + duration + travel_time_delta
                        
                        if potential_end_time <= resolved_start:
                            # We can fit this case
                            current_time = find_next_available_time(current_time, duration, all_blocked_slots, end_hour)
                            
                            # If we exceed end hour or day changes, break to next resolved case
                            if current_time.date() != day:
                                break
                            
                            # Assign timeslot
                            end_time = current_time + duration
                            timeslot_str = f"({current_time.strftime('%d/%m, %H:%M')} - {end_time.strftime('%H:%M')})"
                            result_timeslots.append((coord, postal, timeslot_str))
                            
                            # Update current time and remove this case from remaining
                            current_time = end_time
                            remaining_open_cases.pop(idx)
                            remaining_open_postals.pop(idx)
                            remaining_inspection_times.pop(idx)
                            
                            # Recalculate shortest path if we have more cases
                            if remaining_open_cases:
                                shortest_route = find_shortest_path(remaining_open_cases, coord)
                                i = 0  # Reset index for new route
                            else:
                                break
                        else:
                            # Can't fit before resolved case, try next case
                            i += 1
                    except ValueError:
                        # Coordinate not found in remaining open cases
                        i += 1
            
            # Add the resolved case to the result
            resolved_postal = f"RC{resolved_idx+1:03}"
            timeslot_str = f"({resolved_start.strftime('%d/%m, %H:%M')} - {resolved_end.strftime('%H:%M')})"
            result_timeslots.append((resolved_loc, resolved_postal, timeslot_str))
            
            # Update current time to after this resolved case
            current_time = resolved_end
    
    # Process remaining open cases that couldn't be scheduled around resolved cases
    if remaining_open_cases:
        # Create route with postal codes for remaining open cases
        remaining_route_with_postal = list(zip(remaining_open_cases, remaining_open_postals))
        
        # Use the original algorithm for remaining cases
        remaining_timeslots = assign_timeslots_stable_with_travel_time(
            route_with_postal=remaining_route_with_postal,
            inspection_times=remaining_inspection_times,
            resolved_cases=[],
            blocked_slots=all_blocked_slots,
            holidays_set=holidays_set,
            current_date=max(resolved_days) if resolved_days else current_date,
            start_hour=start_hour,
            end_hour=end_hour,
            max_distance_km=max_distance_km,
            max_cases_per_slot=max_cases_per_slot
        )
        
        # Add remaining timeslots to result
        result_timeslots.extend(remaining_timeslots)
    
    # Sort result by timeslot
    result_timeslots.sort(key=lambda x: datetime.strptime(x[2].strip("()").split(" - ")[0], "%d/%m, %H:%M"))
    
    return result_timeslots

def assign_timeslots_stable_with_travel_time(route_with_postal, inspection_times, resolved_cases=[], blocked_slots=None, holidays_set=None, current_date=datetime.now().date(), start_hour=9, end_hour=18, max_distance_km=1, max_cases_per_slot=3):
    """
    Assign timeslots for inspection cases along a route, considering travel time, blocked slots, and holidays.
    
    Parameters:
    - route_with_postal: List of tuples (coordinate, postal code) representing the route.
    - inspection_times: List of inspection duration (in hours) corresponding to each case.
    - resolved_cases: List of tuples for resolved cases (start_dt, end_dt, location).
    - blocked_slots: List of blocked time intervals (tuples of start and end datetime).
    - holidays_set: Set of holiday dates.
    - current_date: The current date from which scheduling starts.
    - start_hour: Hour of the day when inspections can start.
    - end_hour: Hour of the day by which inspections must end.
    - max_distance_km: Maximum allowed distance between consecutive cases to group them.
    - max_cases_per_slot: Maximum number of cases allowed in a single timeslot group.
    
    Returns:
    - A list of tuples (coordinate, postal, timeslot_str) where timeslot_str is formatted as "(dd/mm, HH:MM - HH:MM)".
    """
    st.session_state.logs.append(f"Assigning timeslots with travel time for route: {route_with_postal}")
    st.session_state.logs.append(f"Inspection times: {inspection_times}")
    st.session_state.logs.append(f"Resolved cases: {resolved_cases}")
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
# endregion

#region Streamlit
# Initialize session state
if "open_cases" not in st.session_state:
    st.session_state.open_cases = []
if "resolved_cases" not in st.session_state:
    st.session_state.resolved_cases = []
if "show_route" not in st.session_state:
    st.session_state.show_route = False
if "route_data" not in st.session_state:
    st.session_state.route_data = None
if "inspection_times" not in st.session_state:
    st.session_state.inspection_times = {}
if "blocked_slots" not in st.session_state:
    st.session_state.blocked_slots = []
if "current_date" not in st.session_state:
    st.session_state.current_date = []
if "logs" not in st.session_state:
    st.session_state.logs = []
if "add_mode" not in st.session_state:
    st.session_state.add_mode = "open"  # can be "open" or "resolved"
if "pending_resolved" not in st.session_state:
    st.session_state.pending_resolved = None  # to temporarily hold a new resolved case location
# Add to the session state initialization section
if "resolved_case_counter" not in st.session_state:
    st.session_state.resolved_case_counter = 0
if "holiday_set" not in st.session_state:
    st.session_state.holiday_set = set()

# Add this function to create a popup-like experience
def show_logs():
    with st.expander("🐛 Debug Logs", expanded=False):
        if st.button("Clear Logs"):
            st.session_state.logs = []
        log_text = "\n".join(st.session_state.logs)
        st.code(log_text, language="plaintext")

# Streamlit App
st.set_page_config(layout="centered")
st.title("Appointment Scheduling Simulation")

# Add blocked time slots input
st.sidebar.header("Additional Settings")

# Add a mode selector for adding cases
mode = st.sidebar.radio("Select Mode", options=["Add Open Case", "Add Resolved Case"])
st.session_state.add_mode = "open" if mode == "Add Open Case" else "resolved"

blocked_times = st.sidebar.text_input(
    "Enter blocked time slots (format: '26/04/2025 9:30 - 10:30, 30/04/2025 13:00 - 16:30')",
    "07/04/2025 9:30 - 10:30, 08/04/2025 8:00 - 18:00, 30/04/2025 13:00 - 16:30",  # Default value
    help="Input Lunch Breaks, Leaves, Holiday, Appointment. If whole day is blocked, use 'dd/mm/yyyy 9:00 - 18:00'."
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

# Add to the sidebar after the blocked times input
st.sidebar.subheader("Resolved Cases")
if st.session_state.resolved_cases:
    resolved_cases_info = ""
    for case in st.session_state.resolved_cases:
        lat, lng = case["location"]
        resolved_cases_info += (
            f"Case ID: {case['case_id']}\n"
            f"Location: ({lat:.4f}, {lng:.4f})\n"
            f"Timeslot: {case['timeslot']}\n"
            f"-------------------\n"
        )
    st.sidebar.text_area("Cases Information", resolved_cases_info, height=200)
else:
    st.sidebar.text("No resolved cases yet")

# Add Open Cases section to the sidebar
st.sidebar.subheader("Open Cases")
if st.session_state.open_cases:
    open_cases_info = ""
    for i, point in enumerate(st.session_state.open_cases):
        lat, lng = point
        label = f"OC{i+1:03}"
        open_cases_info += (
            f"Case ID: {label}\n"
            f"Location: ({lat:.4f}, {lng:.4f})\n"
            f"-------------------\n"
        )
    st.sidebar.text_area("Open Cases Information", open_cases_info, height=200)
else:
    st.sidebar.text("No open cases yet")

# Create a map
m = folium.Map(location=[1.3521, 103.8198], zoom_start=12)  # Singapore coordinates
Draw(export=True).add_to(m)

# Add the map to the Streamlit app
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    map_data = st_folium(m, width=700, height=500)

# Process map clicks to add points
if map_data["last_clicked"]:
    coords = [map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]]
    
    if st.session_state.add_mode == "open":
        # Add to open cases
        if coords not in st.session_state.open_cases:
            st.session_state.open_cases.append(tuple(coords))
            st.success(f"✅ Open case added at coordinates: {coords}")
    else:
        # For resolved cases, we need to capture the location first, then ask for a timeslot
        st.session_state.pending_resolved = tuple(coords)
        st.session_state.logs.append(f"Pending resolved case at: {coords}")

# If we have a pending resolved case, show the timeslot input
if st.session_state.pending_resolved:
    with st.form("resolved_case_form"):
        st.write(f"Adding resolved case at: {st.session_state.pending_resolved}")
        timeslot = st.text_input("Enter timeslot (format: '26/04, 9:30 - 10:30')", "08/04, 14:00 - 15:00")
        
        submit_button = st.form_submit_button("Add Resolved Case")
        if submit_button:
            # Increment the counter for unique case IDs
            st.session_state.resolved_case_counter += 1
            
            # Add to resolved cases
            st.session_state.resolved_cases.append({
                "case_id": f"RC{st.session_state.resolved_case_counter:03}",
                "location": st.session_state.pending_resolved,
                "timeslot": f"({timeslot})"
            })
            
            # Clear the pending resolved case
            st.session_state.pending_resolved = None
            st.success(f"✅ Resolved case added with timeslot: {timeslot}")
            st.rerun()

# Parse blocked times and holidays
st.session_state.blocked_slots = parse_blocked_times(blocked_times)
st.session_state.holiday_set = parse_holidays(holidays_str)

# Add a button to calculate and display the route
with col1:
    if st.button("🗺️ Calculate Route"):
        if not st.session_state.open_cases:
            st.error("⚠️ Please add at least one open case first")
        else:
            try:
                # Parse resolved cases for the algorithm
                parsed_resolved_cases = []
                for case in st.session_state.resolved_cases:
                    try:
                        timeslot = case["timeslot"]
                        # Extract date and time parts
                        timeslot = timeslot.strip("()")
                        date_part, time_part = timeslot.split(", ")
                        start_time_str, end_time_str = time_part.split(" - ")
                        
                        # Parse date (assuming current year)
                        day, month = map(int, date_part.split("/"))
                        current_year = datetime.now().year
                        date_obj = datetime(current_year, month, day)
                        
                        # Parse start and end times
                        start_h, start_m = map(int, start_time_str.split(":"))
                        end_h, end_m = map(int, end_time_str.split(":"))
                        
                        # Create full datetime objects
                        start_datetime = date_obj.replace(hour=start_h, minute=start_m)
                        end_datetime = date_obj.replace(hour=end_h, minute=end_m)
                        
                        parsed_resolved_cases.append((start_datetime, end_datetime, case["location"]))
                    except Exception as e:
                        st.error(f"Error parsing resolved case: {str(e)}")
                        st.session_state.logs.append(f"Error parsing resolved case: {str(e)}")
                
                # If we have open cases, calculate the route
                if not st.session_state.resolved_cases:
                    # If no resolved cases, use the first open case as starting point
                    startCoords = st.session_state.open_cases[0]
                    shortest_route = find_shortest_path(st.session_state.open_cases, startCoords)
                    route_with_postal = [(coord, f"OC{i+1:03}") for i, coord in enumerate(shortest_route)]
                    
                    # Get inspection time inputs (ensure they retain the previous values)
                    inspection_times = []
                    for i in range(len(st.session_state.open_cases)):
                        inspection_time = st.session_state.inspection_times.get(i, 1)  # Default to 1 if not set
                        inspection_times.append(inspection_time)
                    
                    # When calling assign_timeslots_stable_with_travel_time, add blocked_slots parameter:
                    route_with_timeslot = assign_timeslots_stable_with_travel_time(
                        route_with_postal=route_with_postal,
                        inspection_times=inspection_times,
                        resolved_cases=parsed_resolved_cases,
                        blocked_slots=st.session_state.blocked_slots,
                        holidays_set=st.session_state.holiday_set,
                        current_date=st.session_state.current_date
                    )
                else:
                    # If we have resolved cases, use the new algorithm
                    # First, create a list of open cases with postal codes
                    open_cases_with_postal = [(coord, f"OC{i+1:03}") for i, coord in enumerate(st.session_state.open_cases)]
                    
                    # Get inspection time inputs
                    inspection_times = []
                    for i in range(len(st.session_state.open_cases)):
                        inspection_time = st.session_state.inspection_times.get(i, 1)  # Default to 1 if not set
                        inspection_times.append(inspection_time)
                    
                    # Use the new algorithm that considers resolved cases
                    route_with_timeslot = assign_timeslots_with_resolved_cases(
                        open_cases_with_postal=open_cases_with_postal,
                        inspection_times=inspection_times,
                        resolved_cases=st.session_state.resolved_cases,
                        blocked_slots=st.session_state.blocked_slots,
                        holidays_set=st.session_state.holiday_set,
                        current_date=st.session_state.current_date
                    )
                
                st.session_state.route_data = route_with_timeslot
                st.session_state.show_route = True
                st.success("✅ Route calculated successfully!")
                # Redraw the map with the new route
                st.rerun()
            except Exception as e:
                st.error(f"Error calculating route: {str(e)}")
                st.session_state.logs.append(f"Error calculating route: {str(e)}")

# Display the route on the map if available
if st.session_state.show_route and st.session_state.route_data:
    # Create a new map with the route
    route_map = folium.Map(location=[1.3521, 103.8198], zoom_start=12)
    
    # Sort the route data by timeslot
    sorted_route = sorted(
        st.session_state.route_data,
        key=lambda x: datetime.strptime(x[2].strip("()").split(" - ")[0], "%d/%m, %H:%M")
    )
    
    # Add markers for each point in the route
    for i, (coord, postal, timeslot) in enumerate(sorted_route):
        # Determine if this is an open or resolved case
        is_resolved = postal.startswith("RC")
        
        # Choose color based on case type
        color = "red" if is_resolved else "blue"
        
        # Create a popup with information
        popup_text = f"{postal}<br>{timeslot}"
        popup = folium.Popup(popup_text, max_width=300)
        
        # Add a marker with the appropriate color
        folium.Marker(
            location=coord,
            popup=popup,
            icon=folium.Icon(color=color, icon="info-sign"),
        ).add_to(route_map)
        
        # Add a number label
        folium.map.Marker(
            coord,
            icon=folium.DivIcon(
                icon_size=(150,36),
                icon_anchor=(0,0),
                html=f'<div style="font-size: 12pt; color: white; background-color: {color}; border-radius: 50%; width: 20px; height: 20px; text-align: center;">{i+1}</div>',
            )
        ).add_to(route_map)
    
    # Add lines connecting the points in order
    route_points = [coord for coord, _, _ in sorted_route]
    if route_points:
        folium.PolyLine(
            route_points,
            color="blue",
            weight=2,
            opacity=0.7,
            dash_array="5, 5",
        ).add_to(route_map)
    
    # Display the route map
    with col1:
        st.write("### Optimized Route")
        st_folium(route_map, width=700, height=500)
        
        # Display the route information
        st.write("### Route Details")
        for i, (coord, postal, timeslot) in enumerate(sorted_route):
            st.write(f"{i+1}. {postal}: {timeslot} at {coord[0]:.4f}, {coord[1]:.4f}")

with col2:
    if st.button("🔄 Reset Open Cases"):
        st.session_state.open_cases = []
        st.session_state.show_route = False
        st.session_state.route_data = None
        st.session_state.inspection_times = {}
        st.rerun()

with col3:
    if st.button("🔄 Reset Resolved Cases"):
        st.session_state.resolved_cases = []
        st.success("Resolved cases reset successfully.")
        st.rerun()

# Display current open_cases (with correct order based on the route)
if st.session_state.open_cases:
    st.session_state.logs.append("Current open_cases:")
    for i, point in enumerate(st.session_state.open_cases):
        lat, lng = point
        # Generate the label based on the current index
        label = f"OC{i+1:03}"
        st.session_state.logs.append(f"Point {label}: Latitude {lat:.4f}, Longitude {lng:.4f}")
        
        # Add an inspection time input for each point
        inspection_time = st.number_input(f"Inspection Time for Case {i+1}: {label}", min_value=1, max_value=8, value=st.session_state.inspection_times.get(i, 1), key=f"inspection_time_{i}")
        st.session_state.inspection_times[i] = inspection_time  # Store the input value
        
        # Add a delete button for each point
        delete_button = st.button(f"❌ Delete {label}", key=f"delete_{i}")
        if delete_button:
            st.session_state.open_cases.pop(i)
            st.session_state.show_route = False  # Reset the route when a point is deleted
            st.session_state.route_data = None  # Clear the route data
            st.success(f"✅ Case {label} deleted successfully.")
            #break  # Break to avoid mutating the list while iterating over it

if st.session_state.resolved_cases:
    st.subheader("Resolved Cases")
    for i, case in enumerate(st.session_state.resolved_cases):
        loc = case["location"]
        ts = case["timeslot"]
        st.write(f"Resolved Case {i+1}: Location {loc}, Timeslot {ts}")
        delete_resolved = st.button(f"❌ Delete Resolved Case {i+1}", key=f"delete_resolved_{i}")
        if delete_resolved:
            st.session_state.resolved_cases.pop(i)
            st.success(f"Resolved Case {i+1} deleted successfully.")

# Show debug logs at the bottom
show_logs()
#endregion
