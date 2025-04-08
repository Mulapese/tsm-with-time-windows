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

def assign_timeslots_stable_with_travel_time(route_with_postal, inspection_times, resolved_cases=[], blocked_slots=None, holidays_set=None, current_date=datetime.now().date(), start_hour=9, end_hour=18, max_distance_km=1, max_cases_per_slot=3):
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
    
    # If no resolved cases, fall back to the existing algorithm
    if not resolved_cases:
        st.session_state.logs.append("No resolved cases, falling back to original algorithm")
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
                
                for case in ordered_cases:
                    if not can_fit_more:
                        break
                        
                    # Calculate travel time from previous location
                    travel_time_min = calculate_travel_time_km(last_location, case[0])
                    travel_duration = timedelta(minutes=math.ceil(travel_time_min / 30) * 30)
                    
                    # Calculate earliest possible start for this case
                    earliest_start = last_end_time + travel_duration
                    earliest_start = find_earliest_start_time(earliest_start, 
                                                             timedelta(hours=inspection_times_map[case[0]]), 
                                                             blocked_slots)
                    
                    # Calculate end time
                    end_time = earliest_start + timedelta(hours=inspection_times_map[case[0]])
                    
                    # Check if we can still fit this before first resolved case
                    # We need to allow travel time to the resolved case as well
                    travel_to_resolved = calculate_travel_time_km(case[0], first_resolved["location"])
                    travel_to_resolved_duration = timedelta(minutes=math.ceil(travel_to_resolved / 30) * 30)
                    
                    if end_time + travel_to_resolved_duration <= first_resolved["start_time"]:
                        time_slot = f"({earliest_start.strftime('%d/%m, %H:%M')} - {end_time.strftime('%H:%M')})"
                        cases_that_fit.append((case[0], case[1], time_slot))
                        open_cases_before.append(case)
                        last_end_time = end_time
                        last_location = case[0]
                    else:
                        can_fit_more = False
                
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
                    
                    min_required_time = (timedelta(minutes=math.ceil(travel_from_current / 30) * 30) + 
                                         inspection_duration + 
                                         timedelta(minutes=math.ceil(travel_to_next / 30) * 30))
                    
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
                    
                    for case in ordered_cases:
                        # Calculate travel time from previous location
                        travel_time_min = calculate_travel_time_km(last_location, case[0])
                        travel_duration = timedelta(minutes=math.ceil(travel_time_min / 30) * 30)
                        
                        # Calculate earliest possible start
                        earliest_start = current_time + travel_duration
                        earliest_start = find_earliest_start_time(earliest_start, 
                                                                timedelta(hours=inspection_times_map[case[0]]), 
                                                                blocked_slots)
                        
                        # Calculate end time
                        end_time = earliest_start + timedelta(hours=inspection_times_map[case[0]])
                        
                        # Calculate travel time to next resolved
                        travel_to_next = calculate_travel_time_km(case[0], end_coord)
                        travel_to_next_duration = timedelta(minutes=math.ceil(travel_to_next / 30) * 30)
                        
                        # Check if we can still fit this
                        if end_time + travel_to_next_duration <= available_end:
                            time_slot = f"({earliest_start.strftime('%d/%m, %H:%M')} - {end_time.strftime('%H:%M')})"
                            final_result.append((case[0], case[1], time_slot))
                            cases_between.append(case)
                            current_time = end_time
                            last_location = case[0]
                        else:
                            break
                    
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
                travel_duration = timedelta(minutes=math.ceil(travel_time_min / 30) * 30)
                current_time += travel_duration
            
            # Find next available slot after this time
            first_available = find_earliest_start_time(current_time, timedelta(hours=1), combined_blocked)
            
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
                max_cases_per_slot=max_cases_per_slot
            )
            
            final_result.extend(remaining_slots)
        # Handle case where no resolved cases exist or all open cases still need scheduling
        if not parsed_resolved_cases or (len(remaining_open_cases) == len(open_cases_with_postal)):
            # Use original algorithm for remaining cases
            st.session_state.logs.append(f"Scheduling {len(remaining_open_cases)} remaining open cases with original algorithm")
            
            route_with_postal = remaining_open_cases
            inspection_times_for_remaining = [inspection_times_map[case[0]] for case in remaining_open_cases]
            
            remaining_slots = assign_timeslots_stable_with_travel_time(
                route_with_postal=route_with_postal,
                inspection_times=inspection_times_for_remaining,
                resolved_cases=[],
                blocked_slots=blocked_slots,
                holidays_set=holidays_set,
                current_date=current_date,
                start_hour=start_hour,
                end_hour=end_hour,
                max_distance_km=max_distance_km,
                max_cases_per_slot=max_cases_per_slot
            )
            
            # Add these to the final result
            for slot in remaining_slots:
                if slot not in final_result:
                    final_result.append(slot)
    
    # Sort the final result by timeslot
    final_result.sort(key=get_slot_time)
    
    return final_result

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
