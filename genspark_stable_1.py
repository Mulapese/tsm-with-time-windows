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
    # Calculate travel time in minutes based on coordinates
    # Using 10 minutes per km as a baseline rate
    distance_km = geodesic(coord1, coord2).km
    travel_time_minutes = distance_km * 10  # 10 minutes per km
    
    # Round up to nearest 30 minute increment
    rounded_minutes = math.ceil(travel_time_minutes / 30) * 30
    
    return rounded_minutes

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

def assign_timeslots_stable_with_travel_time(route_with_postal, inspection_times, resolved_cases=[], blocked_slots=None, holidays_set=None, current_date=datetime.now().date(), start_hour=9, end_hour=18, max_distance_km=1, max_cases_per_slot=3, skip_working_days_addition=False):
    """
    Assign timeslots for inspection cases along a route, considering travel time, blocked slots, and holidays.
    
    Parameters:
    - route_with_postal: List of tuples (coordinate, postal code) representing the route.
    - inspection_times: List of inspection duration (in hours) corresponding to each case.
    - parsed_resolved_cases:  List of tuples for resolved cases (start_dt, end_dt, location).
    - blocked_slots: List of blocked time intervals (tuples of start and end datetime).
    - holidays_set: Set of holiday dates.
    - current_date: The current date from which scheduling starts.
    - start_hour: Hour of the day when inspections can start.
    - end_hour: Hour of the day by which inspections must end.
    - max_distance_km: Maximum allowed distance between consecutive cases to group them.
    - max_cases_per_slot: Maximum number of cases allowed in a single timeslot group.
    - skip_working_days_addition: If True, skip adding 2 working days to the start date.
    
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
    if skip_working_days_addition:
        start_date = current_date
        st.session_state.logs.append(f"Skipping 2 working days addition, using date: {start_date}")
    else:
        start_date = add_working_days(current_date, 2, holidays_set)
        st.session_state.logs.append(f"Adding 2 working days, resulting date: {start_date}")
    
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
                rounded_travel_time = timedelta(minutes=travel_time_min)
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
                    rounded_travel_time = timedelta(minutes=travel_time_min)
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
                        rounded_travel_time = timedelta(minutes=travel_time_min)
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

def parse_resolved_case_timeslot(timeslot_str):
    """Parse a timeslot string like '11/04/2025 11:00 - 12:00' into datetime objects"""
    try:
        date_part, time_part = timeslot_str.split(" ", 1)
        start_time_str, end_time_str = time_part.split(" - ")
        
        date_obj = datetime.strptime(date_part, "%d/%m/%Y").date()
        start_time = datetime.strptime(start_time_str, "%H:%M").time()
        end_time = datetime.strptime(end_time_str, "%H:%M").time()
        
        start_datetime = datetime.combine(date_obj, start_time)
        end_datetime = datetime.combine(date_obj, end_time)
        
        return start_datetime, end_datetime
    except Exception as e:
        st.session_state.logs.append(f"Error parsing timeslot '{timeslot_str}': {e}")
        return None, None

def assign_timeslots_with_resolved_cases(open_cases_with_postal, inspection_times, resolved_cases, blocked_slots=None, 
                                         holidays_set=None, current_date=datetime.now().date(), 
                                         start_hour=9, end_hour=18, max_distance_km=1, max_cases_per_slot=3):
    """
    Assign timeslots for open cases while respecting fixed resolved case timeslots.
    
    Parameters:
    - open_cases_with_postal: List of tuples (coordinate, postal code) for open cases
    - inspection_times: List of inspection duration (in hours) corresponding to each open case
    - resolved_cases: List of dictionaries with resolved case information
    - blocked_slots: List of blocked time intervals (tuples of start and end datetime)
    - holidays_set: Set of holiday dates
    - current_date: The current date from which scheduling starts
    - start_hour: Hour of the day when inspections can start
    - end_hour: Hour of the day by which inspections must end
    - max_distance_km: Maximum allowed distance between consecutive cases to group them
    - max_cases_per_slot: Maximum number of cases allowed in a single timeslot group
    
    Returns:
    - A list of tuples (coordinate, postal, timeslot_str) containing both open and resolved cases
    """
    st.session_state.logs.append(f"Starting integrated scheduling with {len(open_cases_with_postal)} open cases and {len(resolved_cases)} resolved cases")
    
    if blocked_slots is None:
        blocked_slots = []
    
    # If no resolved cases, use shortest path algorithm for open cases
    if not resolved_cases:
        st.session_state.logs.append("No resolved cases, using shortest path algorithm for open cases")
        
        # Check if we have open cases
        if open_cases_with_postal:
            # Extract coordinates for routing
            coords_only = [case[0] for case in open_cases_with_postal]
            
            # Start with the first case by default
            start_coord = coords_only[0]
            
            # Find optimal route using shortest path
            optimal_route = find_shortest_path(coords_only, start_coord)
            
            # Create a mapping from coordinates to the full case info
            coord_to_case = {case[0]: case for case in open_cases_with_postal}
            
            # Reorder open cases and inspection times based on the optimal route
            reordered_open_cases = [coord_to_case[coord] for coord in optimal_route]
            reordered_inspection_times = []
            
            # Create a mapping from original coordinates to inspection times
            coord_to_time = {open_cases_with_postal[i][0]: inspection_times[i] for i in range(len(open_cases_with_postal))}
            
            # Reorder inspection times to match the new route order
            for case in reordered_open_cases:
                reordered_inspection_times.append(coord_to_time[case[0]])
            
            st.session_state.logs.append(f"Reordered {len(reordered_open_cases)} open cases using shortest path algorithm")
            
            return assign_timeslots_stable_with_travel_time(
                route_with_postal=reordered_open_cases,
                inspection_times=reordered_inspection_times,
                resolved_cases=[],
                blocked_slots=blocked_slots,
                holidays_set=holidays_set,
                current_date=current_date,
                start_hour=start_hour,
                end_hour=end_hour,
                max_distance_km=max_distance_km,
                max_cases_per_slot=max_cases_per_slot,
                skip_working_days_addition=False  # Apply the default 2 working days for initial scheduling
            )
        else:
            # If no open cases, just return an empty list
            return []
    
    # Parse resolved cases to get datetime objects
    parsed_resolved_cases = []
    for case in resolved_cases:
        start_dt, end_dt = parse_resolved_case_timeslot(case["timeslot"])
        if start_dt and end_dt:
            parsed_resolved_cases.append({
                "case_id": case["case_id"],
                "location": case["location"],
                "start_time": start_dt,
                "end_time": end_dt
            })
    
    # Sort resolved cases by start_time
    parsed_resolved_cases.sort(key=lambda x: x["start_time"])
    
    # Mark resolved case timeslots as blocked
    resolved_blocked_slots = blocked_slots.copy()
    for case in parsed_resolved_cases:
        resolved_blocked_slots.append((case["start_time"], case["end_time"]))
    
    # Initialize result container
    final_result = []
    
    # Define minimum start date (2 working days from current date)
    min_start_date = add_working_days(current_date, 2, holidays_set)
    base_time = datetime.combine(min_start_date, time(start_hour, 0))
    
    # If we have resolved cases, create a combined route with resolved cases as anchors
    integrated_route = []
    remaining_open_cases = list(open_cases_with_postal)
    inspection_times_map = {open_cases_with_postal[i][0]: inspection_times[i] for i in range(len(open_cases_with_postal))}
    
    # Helper function to get key value for sorting slots
    def get_slot_time(slot_tuple):
        if isinstance(slot_tuple, tuple) and len(slot_tuple) >= 3:
            if isinstance(slot_tuple[2], str) and "(" in slot_tuple[2]:
                # Extract dates and times from slot string format "(dd/mm, HH:MM - HH:MM)"
                slot_str = slot_tuple[2]
                date_part = slot_str.split(",")[0].strip("(")
                time_part = slot_str.split(",")[1].split("-")[0].strip()
                try:
                    # Create a datetime for sorting
                    date_obj = datetime.strptime(f"{date_part}/2025", "%d/%m/%Y").date()
                    time_obj = datetime.strptime(time_part, " %H:%M").time()
                    return datetime.combine(date_obj, time_obj)
                except:
                    return datetime.max
        return datetime.max  # Default for any format issues
    
    # Helper function to find earliest available start time
    def find_earliest_start_time(current_time, duration, blocked_slots):
        original_time = current_time
        while is_time_blocked(current_time, duration, blocked_slots) or exceeds_end_hour(current_time, duration):
            current_time += timedelta(minutes=30)
            if current_time.hour >= end_hour or exceeds_end_hour(current_time, duration):
                next_day = current_time.date() + timedelta(days=1)
                while next_day.weekday() >= 5 or next_day in holidays_set:
                    next_day += timedelta(days=1)
                current_time = datetime.combine(next_day, time(start_hour, 0))
        
        st.session_state.logs.append(f"Earliest start time from {original_time} with duration {duration}: {current_time}")
        return current_time
    
    def exceeds_end_hour(start_time, duration):
        end_time = start_time + duration
        return end_time.hour > end_hour or (end_time.hour == end_hour and end_time.minute > 0)
    
    # Add resolved cases to the final result first
    for resolved_case in parsed_resolved_cases:
        time_slot = f"({resolved_case['start_time'].strftime('%d/%m, %H:%M')} - {resolved_case['end_time'].strftime('%H:%M')})"
        final_result.append((resolved_case["location"], resolved_case["case_id"], time_slot))
    
    # If there are open cases to schedule
    if remaining_open_cases:
        # Split scheduling into segments - before first resolved, between resolved cases, after last resolved
        
        # 1. Schedule open cases before the first resolved case (if any)
        if parsed_resolved_cases:
            first_resolved = parsed_resolved_cases[0]
            available_end_time = first_resolved["start_time"]
            
            # Figure out how many open cases we can fit before the first resolved case
            open_cases_before = []
            current_time = base_time
            
            # Try to find a good starting point for optimization
            if remaining_open_cases:
                # Find case closest to the first resolved case as our anchor
                start_coord = min(remaining_open_cases, key=lambda x: geodesic(x[0], first_resolved["location"]).km)[0]
                
                # Find optimal route through remaining cases starting from this point
                coords_only = [x[0] for x in remaining_open_cases]
                optimal_route = find_shortest_path(coords_only, start_coord)
                
                # Map back to full case info
                coord_to_case = {x[0]: x for x in remaining_open_cases}
                ordered_cases = [coord_to_case[coord] for coord in optimal_route]
                
                # Calculate how many cases we can fit before first resolved
                last_end_time = current_time
                last_location = ordered_cases[0][0]
                can_fit_more = True
                cases_that_fit = []
                
                # Initialize grouping variables
                current_group = []
                current_group_distance = 0
                group_time = timedelta(hours=1)  # Standard 1-hour slot for grouped cases
                
                for case in ordered_cases:
                    if not can_fit_more:
                        break
                        
                    inspection_time = inspection_times_map[case[0]]
                    
                    # If inspection time > 1, handle as individual appointment
                    if inspection_time > 1:
                        # If we have a pending group, schedule it first
                        if current_group:
                            # Find suitable time for the group
                            travel_time_min = calculate_travel_time_km(last_location, current_group[0][0])
                            travel_duration = timedelta(minutes=travel_time_min)
                            earliest_group_start = find_earliest_start_time(current_time + travel_duration, 
                                                                           group_time, 
                                                                           blocked_slots)
                            group_end_time = earliest_group_start + group_time
                            
                            # Check if group still fits before next resolved case
                            travel_to_next = calculate_travel_time_km(current_group[-1][0], first_resolved["location"])
                            travel_duration = timedelta(minutes=travel_to_next)
                            
                            if group_end_time + travel_duration <= first_resolved["start_time"]:
                                time_slot = f"({earliest_group_start.strftime('%d/%m, %H:%M')} - {group_end_time.strftime('%H:%M')})"
                                for group_case in current_group:
                                    cases_that_fit.append((group_case[0], group_case[1], time_slot))
                                    open_cases_before.append(group_case)
                                
                                current_time = group_end_time
                                last_location = current_group[-1][0]
                            else:
                                can_fit_more = False
                                break
                        
                            # Reset group
                            current_group = []
                            current_group_distance = 0
                        
                        # Calculate travel time from previous location
                        travel_time_min = calculate_travel_time_km(last_location, case[0])
                        travel_duration = timedelta(minutes=travel_time_min)
                        
                        # Calculate earliest possible start for this case
                        earliest_start = current_time + travel_duration
                        earliest_start = find_earliest_start_time(earliest_start, 
                                                                timedelta(hours=inspection_time), 
                                                                blocked_slots)
                        
                        # Calculate end time
                        end_time = earliest_start + timedelta(hours=inspection_time)
                        
                        # Calculate travel time to next resolved
                        travel_to_next = calculate_travel_time_km(case[0], first_resolved["location"])
                        travel_to_next_duration = timedelta(minutes=travel_to_next)
                        
                        # Check if we can still fit this
                        if end_time + travel_to_next_duration <= first_resolved["start_time"]:
                            time_slot = f"({earliest_start.strftime('%d/%m, %H:%M')} - {end_time.strftime('%H:%M')})"
                            cases_that_fit.append((case[0], case[1], time_slot))
                            open_cases_before.append(case)
                            current_time = end_time
                            last_location = case[0]
                        else:
                            can_fit_more = False
                    else:
                        # Handle 1-hour cases with grouping
                        if not current_group:
                            # Start a new group
                            current_group.append(case)
                            current_group_distance = 0
                        else:
                            # Check if this case can be added to current group
                            last_case_in_group = current_group[-1]
                            distance_km = geodesic(last_case_in_group[0], case[0]).km
                            
                            if distance_km < max_distance_km and len(current_group) < max_cases_per_slot:
                                # Add to current group
                                current_group.append(case)
                                current_group_distance = max(current_group_distance, distance_km)
                            else:
                                # Current group is full or too spread out, schedule it
                                travel_time_min = calculate_travel_time_km(last_location, current_group[0][0])
                                travel_duration = timedelta(minutes=travel_time_min)
                                earliest_group_start = find_earliest_start_time(current_time + travel_duration, 
                                                                              group_time, 
                                                                              blocked_slots)
                                group_end_time = earliest_group_start + group_time
                                
                                # Check if group still fits before next resolved case
                                travel_to_next = calculate_travel_time_km(current_group[-1][0], first_resolved["location"])
                                travel_duration = timedelta(minutes=travel_to_next)
                                
                                if group_end_time + travel_duration <= first_resolved["start_time"]:
                                    time_slot = f"({earliest_group_start.strftime('%d/%m, %H:%M')} - {group_end_time.strftime('%H:%M')})"
                                    for group_case in current_group:
                                        cases_that_fit.append((group_case[0], group_case[1], time_slot))
                                        open_cases_before.append(group_case)
                                    
                                    current_time = group_end_time
                                    last_location = current_group[-1][0]
                                    
                                    # Start a new group with current case
                                    current_group = [case]
                                    current_group_distance = 0
                                else:
                                    break
                
                # Don't forget to schedule the last group if there is one
                if current_group:
                    travel_time_min = calculate_travel_time_km(last_location, current_group[0][0])
                    travel_duration = timedelta(minutes=travel_time_min)
                    earliest_group_start = find_earliest_start_time(current_time + travel_duration, 
                                                                  group_time, 
                                                                  blocked_slots)
                    group_end_time = earliest_group_start + group_time
                    
                    # Check if group still fits before next resolved case
                    travel_to_next = calculate_travel_time_km(current_group[-1][0], first_resolved["location"])
                    travel_duration = timedelta(minutes=travel_to_next)
                    
                    if group_end_time + travel_duration <= first_resolved["start_time"]:
                        time_slot = f"({earliest_group_start.strftime('%d/%m, %H:%M')} - {group_end_time.strftime('%H:%M')})"
                        for group_case in current_group:
                            cases_that_fit.append((group_case[0], group_case[1], time_slot))
                            open_cases_before.append(group_case)
                
                # Add all cases that fit to final result
                final_result.extend(cases_that_fit)
                
                # Remove scheduled cases from remaining
                for case in open_cases_before:
                    remaining_open_cases.remove(case)
        
        # 2. Schedule cases between resolved cases
        for i in range(len(parsed_resolved_cases) - 1):
            current_resolved = parsed_resolved_cases[i]
            next_resolved = parsed_resolved_cases[i + 1]
            
            available_start = current_resolved["end_time"]
            available_end = next_resolved["start_time"]
            
            if remaining_open_cases:
                # Find optimal route between these two resolved points
                start_coord = current_resolved["location"]
                end_coord = next_resolved["location"]
                
                # Filter cases that could potentially fit between these resolved cases
                potential_cases = []
                for case in remaining_open_cases:
                    # Check if case can be reached from current resolved and still make it to next resolved
                    travel_from_current = calculate_travel_time_km(start_coord, case[0])
                    travel_to_next = calculate_travel_time_km(case[0], end_coord)
                    inspection_duration = timedelta(hours=inspection_times_map[case[0]])
                    
                    min_required_time = (timedelta(minutes=travel_from_current) + 
                                         inspection_duration + 
                                         timedelta(minutes=travel_to_next))
                    
                    if available_start + min_required_time <= available_end:
                        potential_cases.append(case)
                
                cases_between = []
                if potential_cases:
                    # Find optimal route through potential cases
                    coords_only = [x[0] for x in potential_cases]
                    
                    # Try to add start_coord if not already in coords_only
                    if start_coord not in coords_only:
                        optimal_route = find_shortest_path(coords_only, min(coords_only, key=lambda x: geodesic(x, start_coord).km))
                    else:
                        optimal_route = find_shortest_path(coords_only, start_coord)
                    
                    # Map back to full case info
                    coord_to_case = {x[0]: x for x in potential_cases}
                    ordered_cases = [coord_to_case.get(coord) for coord in optimal_route if coord in coord_to_case]
                    
                    # Schedule these cases
                    current_time = available_start
                    last_location = start_coord
                    
                    # Initialize grouping variables
                    current_group = []
                    current_group_distance = 0
                    group_time = timedelta(hours=1)  # Standard 1-hour slot for grouped cases
                    
                    for case in ordered_cases:
                        inspection_time = inspection_times_map[case[0]]
                        
                        # If inspection time > 1, handle as individual appointment
                        if inspection_time > 1:
                            # If we have a pending group, schedule it first
                            if current_group:
                                # Find suitable time for the group
                                travel_time_min = calculate_travel_time_km(last_location, current_group[0][0])
                                travel_duration = timedelta(minutes=travel_time_min)
                                earliest_group_start = find_earliest_start_time(current_time + travel_duration, 
                                                                               group_time, 
                                                                               blocked_slots)
                                group_end_time = earliest_group_start + group_time
                                
                                # Check if group still fits before next resolved case
                                travel_to_next = calculate_travel_time_km(current_group[-1][0], end_coord)
                                travel_duration = timedelta(minutes=travel_to_next)
                                
                                if group_end_time + travel_duration <= available_end:
                                    time_slot = f"({earliest_group_start.strftime('%d/%m, %H:%M')} - {group_end_time.strftime('%H:%M')})"
                                    for group_case in current_group:
                                        final_result.append((group_case[0], group_case[1], time_slot))
                                        cases_between.append(group_case)
                                    
                                    current_time = group_end_time
                                    last_location = current_group[-1][0]
                                else:
                                    break
                                
                                # Reset group
                                current_group = []
                                current_group_distance = 0
                            
                            # Calculate travel time from previous location
                            travel_time_min = calculate_travel_time_km(last_location, case[0])
                            travel_duration = timedelta(minutes=travel_time_min)
                            
                            # Calculate earliest possible start for this case
                            earliest_start = current_time + travel_duration
                            earliest_start = find_earliest_start_time(earliest_start, 
                                                                    timedelta(hours=inspection_time), 
                                                                    blocked_slots)
                            
                            # Calculate end time
                            end_time = earliest_start + timedelta(hours=inspection_time)
                            
                            # Calculate travel time to next resolved
                            travel_to_next = calculate_travel_time_km(case[0], end_coord)
                            travel_to_next_duration = timedelta(minutes=travel_to_next)
                            
                            # Check if we can still fit this
                            if end_time + travel_to_next_duration <= available_end:
                                time_slot = f"({earliest_start.strftime('%d/%m, %H:%M')} - {end_time.strftime('%H:%M')})"
                                final_result.append((case[0], case[1], time_slot))
                                cases_between.append(case)
                                current_time = end_time
                                last_location = case[0]
                            else:
                                break
                        else:
                            # Handle 1-hour cases with grouping
                            if not current_group:
                                # Start a new group
                                current_group.append(case)
                                current_group_distance = 0
                            else:
                                # Check if this case can be added to current group
                                last_case_in_group = current_group[-1]
                                distance_km = geodesic(last_case_in_group[0], case[0]).km
                                
                                if distance_km < max_distance_km and len(current_group) < max_cases_per_slot:
                                    # Add to current group
                                    current_group.append(case)
                                    current_group_distance = max(current_group_distance, distance_km)
                                else:
                                    # Current group is full or too spread out, schedule it
                                    travel_time_min = calculate_travel_time_km(last_location, current_group[0][0])
                                    travel_duration = timedelta(minutes=travel_time_min)
                                    earliest_group_start = find_earliest_start_time(current_time + travel_duration, 
                                                                                  group_time, 
                                                                                  blocked_slots)
                                    group_end_time = earliest_group_start + group_time
                                    
                                    # Check if group still fits before next resolved case
                                    travel_to_next = calculate_travel_time_km(current_group[-1][0], end_coord)
                                    travel_duration = timedelta(minutes=travel_to_next)
                                    
                                    if group_end_time + travel_duration <= available_end:
                                        time_slot = f"({earliest_group_start.strftime('%d/%m, %H:%M')} - {group_end_time.strftime('%H:%M')})"
                                        for group_case in current_group:
                                            final_result.append((group_case[0], group_case[1], time_slot))
                                            cases_between.append(group_case)
                                        
                                        current_time = group_end_time
                                        last_location = current_group[-1][0]
                                        
                                        # Start a new group with current case
                                        current_group = [case]
                                        current_group_distance = 0
                                    else:
                                        break
                    
                    # Don't forget to schedule the last group if there is one
                    if current_group:
                        travel_time_min = calculate_travel_time_km(last_location, current_group[0][0])
                        travel_duration = timedelta(minutes=travel_time_min)
                        earliest_group_start = find_earliest_start_time(current_time + travel_duration, 
                                                                      group_time, 
                                                                      blocked_slots)
                        group_end_time = earliest_group_start + group_time
                        
                        # Check if group still fits before next resolved case
                        travel_to_next = calculate_travel_time_km(current_group[-1][0], end_coord)
                        travel_duration = timedelta(minutes=travel_to_next)
                        
                        if group_end_time + travel_duration <= available_end:
                            time_slot = f"({earliest_group_start.strftime('%d/%m, %H:%M')} - {group_end_time.strftime('%H:%M')})"
                            for group_case in current_group:
                                final_result.append((group_case[0], group_case[1], time_slot))
                                cases_between.append(group_case)
                    
                    # Remove scheduled cases from remaining
                    for case in cases_between:
                        if case in remaining_open_cases:
                            remaining_open_cases.remove(case)
        
        # 3. Schedule cases after the last resolved case
        if parsed_resolved_cases and remaining_open_cases:
            last_resolved = parsed_resolved_cases[-1]
            start_time = last_resolved["end_time"]
            start_coord = last_resolved["location"]
            
            # Find optimal route starting from last resolved case
            coords_only = [x[0] for x in remaining_open_cases]
            nearest_to_start = min(coords_only, key=lambda x: geodesic(x, start_coord).km)
            optimal_route = find_shortest_path(coords_only, nearest_to_start)
            
            # Map back to full case info
            coord_to_case = {x[0]: x for x in remaining_open_cases}
            ordered_cases = [coord_to_case[coord] for coord in optimal_route]
            
            # Now use the existing algorithm to schedule these cases, but starting after the last resolved case
            route_with_postal = [(case[0], case[1]) for case in ordered_cases]
            inspection_times_for_remaining = [inspection_times_map[case[0]] for case in ordered_cases]
            
            # Create modified blocked slots list to account for resolved cases
            combined_blocked = blocked_slots.copy()
            for rc in parsed_resolved_cases:
                combined_blocked.append((rc["start_time"], rc["end_time"]))
            
            # Use the existing algorithm, but force the start time to be after the last resolved case
            current_time = start_time
            
            # Calculate travel time from last resolved to first case
            if ordered_cases:
                travel_time_min = calculate_travel_time_km(start_coord, ordered_cases[0][0])
                travel_duration = timedelta(minutes=travel_time_min)
                current_time += travel_duration
            
            # Find next available slot after this time
            first_available = find_earliest_start_time(current_time, timedelta(minutes=30), combined_blocked)
            
            # If we've moved to next day, use the standard start hour
            if first_available.date() > current_time.date():
                pass  # The find_earliest_start_time function already handles this
            
            # Adjust the date to ensure we're starting after the resolved case
            remaining_slots = assign_timeslots_stable_with_travel_time(
                route_with_postal=route_with_postal,
                inspection_times=inspection_times_for_remaining,
                resolved_cases=[],
                blocked_slots=combined_blocked,
                holidays_set=holidays_set,
                current_date=first_available.date(),  # Use the date after last resolved
                start_hour=first_available.hour,  # Use the hour after travel from last resolved
                end_hour=end_hour,
                max_distance_km=max_distance_km,
                max_cases_per_slot=max_cases_per_slot,
                skip_working_days_addition=True  # Skip adding 2 working days since we're continuing from a resolved case
            )
            
            final_result.extend(remaining_slots)
        # Handle case where no resolved cases exist or all open cases still need scheduling
        if not parsed_resolved_cases or (len(remaining_open_cases) == len(open_cases_with_postal)):
            # Use original algorithm for remaining cases
            st.session_state.logs.append(f"Scheduling {len(remaining_open_cases)} remaining open cases with original algorithm")
            
            # For unscheduled open cases, use shortest path algorithm
            if remaining_open_cases:
                # Extract coordinates for routing
                coords_only = [case[0] for case in remaining_open_cases]
                
                # Start with the first case by default
                start_coord = coords_only[0]
                
                # Find optimal route using shortest path
                optimal_route = find_shortest_path(coords_only, start_coord)
                
                # Create a mapping from coordinates to the full case info
                coord_to_case = {case[0]: case for case in remaining_open_cases}
                
                # Reorder remaining open cases based on the optimal route
                reordered_open_cases = [coord_to_case[coord] for coord in optimal_route]
                reordered_inspection_times = [inspection_times_map[case[0]] for case in reordered_open_cases]
                
                st.session_state.logs.append(f"Reordered {len(reordered_open_cases)} remaining open cases using shortest path algorithm")
                
                remaining_slots = assign_timeslots_stable_with_travel_time(
                    route_with_postal=reordered_open_cases,
                    inspection_times=reordered_inspection_times,
                    resolved_cases=[],
                    blocked_slots=blocked_slots,
                    holidays_set=holidays_set,
                    current_date=current_date,
                    start_hour=start_hour,
                    end_hour=end_hour,
                    max_distance_km=max_distance_km,
                    max_cases_per_slot=max_cases_per_slot,
                    skip_working_days_addition=False  # Apply the default 2 working days for initial scheduling
                )
                
                # Add these to the final result
                for slot in remaining_slots:
                    if slot not in final_result:
                        final_result.append(slot)
    
    # Sort the final result by timeslot
    final_result.sort(key=get_slot_time)
    
    return final_result
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



# Add this function to create a popup-like experience
def show_logs():
    with st.expander("🐛 Debug Logs", expanded=False):
        if st.button("Clear Logs"):
            st.session_state.logs = []
        log_text = "\n".join(st.session_state.logs)
        st.code(log_text, language="plaintext")

# Streamlit App
st.set_page_config(layout="centered")
st.title("Appoinment Scheduling Simulation")

# Add blocked time slots input
st.sidebar.header("Additional Settings")

# Add a mode selector for adding cases
mode = st.sidebar.radio("Select Mode", options=["Add Open Case", "Add Resolved Case"])
st.session_state.add_mode = "open" if mode == "Add Open Case" else "resolved"

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

# Update these two conditional blocks to always update the session state
st.session_state.blocked_slots = parse_blocked_times(blocked_times) if blocked_times else []
st.session_state.holiday_set = parse_holidays(holidays_str) if holidays else set()


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
        # Generate case ID
        case_id = f"OC{i+1:03}"
        
        # Use index as key for inspection time lookup
        inspection_time = st.session_state.inspection_times.get(i, 1)
        
        # Get the timeslot if available in route_data
        timeslot = "Not scheduled yet"
        if st.session_state.route_data:
            for coord, postal, ts in st.session_state.route_data:
                if (abs(coord[0] - lat) < 0.0001 and abs(coord[1] - lng) < 0.0001 and postal.startswith("OC")):
                    timeslot = ts
                    break
        
        open_cases_info += (
            f"Case ID: {case_id}\n"
            f"Location: ({lat:.4f}, {lng:.4f})\n"
            f"Inspection Time: {inspection_time} hour(s)\n"
            f"Timeslot: {timeslot}\n"
            f"-------------------\n"
        )
    st.sidebar.text_area("Open Cases Information", open_cases_info, height=200)
else:
    st.sidebar.text("No open cases yet")


# Create the base map
if st.session_state.route_data:
    # Show the route map if we have calculated a route (for open cases)
    route_with_timeslot = st.session_state.route_data
    m = folium.Map(location=route_with_timeslot[0][0], zoom_start=13)
    
    # Sort route by timeslot for visualization (chronological order)
    def get_timeslot_datetime(item):
        try:
            ts = item[2]  # Get the timeslot string
            if ts.startswith("("):
                ts = ts[1:]  # Remove opening parenthesis
            date_part, time_part = ts.split(",")
            start_time, _ = time_part.strip().split(" - ")
            dt_str = f"{date_part.strip()}/2025 {start_time.strip()}"
            return datetime.strptime(dt_str, "%d/%m/%Y %H:%M")
        except:
            return datetime.max
    
    # Sort by timeslot for visualization (show chronological route)
    sorted_route = sorted(route_with_timeslot, key=get_timeslot_datetime)
    
    # Draw markers for all cases first
    for i, (coord, postal, timeslot) in enumerate(sorted_route):
        # Determine marker color based on case type
        if postal.startswith("RC"):
            color = "green"  # Resolved cases
        elif i == 0:
            color = "purple"  # Starting point
        elif i == len(sorted_route) - 1:
            color = "orange"  # Ending point
        else:
            color = "blue"  # Intermediate points
        
        folium.Marker(coord, icon=folium.Icon(color=color)).add_to(m)
        folium.map.Marker(
            location=coord,
            icon=folium.DivIcon(
                icon_size=(150, 36),
                html=f'<div style="font-size: 12px; color: black; background-color: white; padding: 2px; border-radius: 4px; border: 1px solid grey;">{postal}<br>{timeslot}</div>'
            )
        ).add_to(m)
    
    # Draw route connections as a single path between consecutive points
    for i in range(len(sorted_route) - 1):
        c1, p1, _ = sorted_route[i]
        c2, p2, _ = sorted_route[i + 1]
        dist_km = geodesic(c1, c2).km
        
        # Calculate travel time with proper rounding to 30 min increments
        travel_time_minutes = calculate_travel_time_km(c1, c2)
        travel_time_hours = travel_time_minutes / 60
        
        midpoint = ((c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2)
        
        # Different line styles for different connections
        if p1.startswith("RC") and p2.startswith("RC"):
            # Connection between resolved cases
            folium.PolyLine([c1, c2], color="green", weight=3, dash_array="5, 10").add_to(m)
        elif p1.startswith("RC") or p2.startswith("RC"):
            # Connection to/from a resolved case
            folium.PolyLine([c1, c2], color="orange", weight=3, dash_array="5, 5").add_to(m)
        else:
            # Connection between open cases
            folium.PolyLine([c1, c2], color="blue", weight=3).add_to(m)
        
        folium.Marker(midpoint, icon=folium.DivIcon(
            html=f'<div style="font-size: 11px; color: black;">{dist_km:.2f} km<br>{travel_time_hours:.1f} hrs</div>'
        )).add_to(m)
    
    # Add drawing tools (for manual modifications if needed)
    Draw(
        export=False,
        draw_options={"polyline": False, "rectangle": False, "circle": False, "circlemarker": False, "polygon": False},
        edit_options={"edit": False, "remove": True}
    ).add_to(m)
else:
    # Show default map if no route is available
    m = folium.Map(location=[1.3521, 103.8198], zoom_start=12)
    Draw(
        export=False,
        draw_options={"polyline": False, "rectangle": False, "circle": False, "circlemarker": False, "polygon": False},
        edit_options={"edit": False, "remove": True}
    ).add_to(m)


# Add markers for resolved_cases (only if route_data doesn't exist yet)
if not st.session_state.route_data:
    for case in st.session_state.resolved_cases:
        loc = case["location"]
        timeslot = case["timeslot"]
        case_id = case["case_id"]
        folium.Marker(loc, icon=folium.Icon(color="green")).add_to(m)
        folium.map.Marker(
            location=loc,
            icon=folium.DivIcon(
                icon_size=(150, 36),
                html=f'<div style="font-size: 12px; color: black; background-color: white; padding: 2px; border-radius: 4px; border: 1px solid grey;">{case_id}<br>{timeslot}</div>'
            )
        ).add_to(m)


# Display the map and handle drawn open_cases
output = st_folium(m, width=700, height=500, key="map", returned_objects=["last_active_drawing", "all_drawings"])

# Process drawn cases depending on the current mode
if output["last_active_drawing"]:
    coords = output["last_active_drawing"]["geometry"]["coordinates"]
    latlng = (coords[1], coords[0])
    if st.session_state.add_mode == "open":
        if latlng not in st.session_state.open_cases:
            st.session_state.open_cases.append(latlng)
            st.session_state.show_route = False  # Reset route when new points are added
    else:  # "resolved" mode
        # Store the pending resolved case location and trigger the modal popup
        st.session_state.pending_resolved = latlng
        st.session_state.logs.append(f"Selected Resolved Case at: {latlng}")

# If there is a pending resolved case, show a popup (using st.modal)
if st.session_state.pending_resolved:
    with st.expander("Enter Resolved Case Timeslot", expanded=True):
        timeslot_input = st.text_input(
            "Enter timeslot (e.g. '20/04/2025 09:00 - 10:00')", 
            value="14/04/2025 11:00 - 12:00",  # Default value
            help="Input timeslot for the resolved case."
        )
        if st.button("Submit Resolved Case"):
            st.session_state.resolved_case_counter += 1
            case_id = f"RC{st.session_state.resolved_case_counter:03}"
            lat, lng = st.session_state.pending_resolved
            resolved_case = {
                "case_id": case_id,
                "location": (lat, lng),
                "latitude": lat,
                "longitude": lng,
                "timeslot": timeslot_input
            }
            st.session_state.resolved_cases.append(resolved_case)
            st.session_state.logs.append(f"Added Resolved Case: {case_id}")
            st.session_state.pending_resolved = None
            st.rerun()

# Recalculate the route and reindex the open_cases based on the route order
if st.session_state.show_route and st.session_state.route_data:
    route_with_timeslot = st.session_state.route_data
    # Reorder open_cases based on the route order
    ordered_open_cases = []
    for coord, postal, _ in route_with_timeslot:
        if postal.startswith("OC"):  # Only include open cases
            if coord not in ordered_open_cases:
                ordered_open_cases.append(coord)
    
    # Update open_cases list if needed
    if ordered_open_cases:
        st.session_state.open_cases = ordered_open_cases  # Update session state to reflect the correct order


show_logs()

# Parse resolved cases into tuples: (start_dt, end_dt, location)
parsed_resolved_cases = []
for case in st.session_state.resolved_cases:
    try:
        # Extract and parse timeslot and location
        slot_str, loc = case["timeslot"], case["location"]
        date_part, times_part = slot_str.split(" ", 1)
        start_time_str, end_time_str = times_part.split(" - ")
        slot_date = datetime.strptime(date_part, "%d/%m/%Y").date()
        start_dt = datetime.combine(slot_date, datetime.strptime(start_time_str, "%H:%M").time())
        end_dt = datetime.combine(slot_date, datetime.strptime(end_time_str, "%H:%M").time())
        parsed_resolved_cases.append((start_dt, end_dt, loc))
        st.session_state.logs.append(f"Parsed resolved slot: {slot_str} -> {start_dt} to {end_dt}, Location: {loc}")
    except ValueError as e:
        st.session_state.logs.append(f"Error parsing resolved slot '{case['timeslot']}': {e}")


# Button actions
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🧮 Calculate Route"):
        # We need at least one case (open or resolved) to calculate a route
        if len(st.session_state.open_cases) == 0 and len(st.session_state.resolved_cases) == 0:
            st.warning("Add at least 1 open case or 1 resolved case.")
        else:
            # Prepare open cases with postal codes
            open_cases_with_postal = [(coord, f"OC{i+1:03}") for i, coord in enumerate(st.session_state.open_cases)]
            
            # Get inspection time inputs (ensure they retain the previous values)
            inspection_times = []
            for i in range(len(st.session_state.open_cases)):
                inspection_time = st.session_state.inspection_times.get(i, 1)  # Default to 1 if not set
                inspection_times.append(inspection_time)
            
            # Call the new integrated scheduling algorithm
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
            # Redraw the map with the new route
            st.rerun()


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
    st.subheader("Open Cases")
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
            st.rerun()  # Rerun to update the UI

if st.session_state.resolved_cases:
    st.subheader("Resolved Cases")
    for i, case in enumerate(st.session_state.resolved_cases):
        loc = case["location"]
        ts = case["timeslot"]
        case_id = case["case_id"]
        st.write(f"{case_id}: Location {loc}, Timeslot {ts}")
        delete_resolved = st.button(f"❌ Delete {case_id}", key=f"delete_resolved_{i}")
        if delete_resolved:
            st.session_state.resolved_cases.pop(i)
            st.session_state.show_route = False  # Reset the route when a case is deleted
            st.session_state.route_data = None  # Clear the route data
            st.success(f"{case_id} deleted successfully.")
            st.rerun()  # Rerun to update the UI
#endregion