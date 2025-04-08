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

def exceeds_end_hour(current_time, duration):
    """Check if a timeslot would exceed the end hour of the workday."""
    end_time = current_time + duration
    return end_time.hour >= 18 or (end_time.hour == 18 and end_time.minute > 0)

def move_to_next_day(current_time):
    """Move the current time to the start of the next working day."""
    next_day = current_time.date() + timedelta(days=1)
    # Skip weekends
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    return datetime.combine(next_day, time(9, 0))

def find_next_available_time(current_time, duration, blocked_slots, end_hour):
    """Find the next available time slot that doesn't overlap with blocked slots."""
    # Ensure we're working with a copy of current_time
    check_time = current_time
    
    # If we're past the end hour, move to the next day
    if check_time.hour >= end_hour:
        check_time = move_to_next_day(check_time)
    
    # Keep checking until we find an available slot
    while True:
        # Check if this slot is blocked
        if is_time_blocked(check_time, duration, blocked_slots):
            # Move forward by 30 minutes
            check_time += timedelta(minutes=30)
            # If we're now past the end hour, move to the next day
            if check_time.hour >= end_hour:
                check_time = move_to_next_day(check_time)
        else:
            # Found an available slot
            return check_time

def can_fit_before_resolved(coord, inspection_time, current_time, next_rc_start, next_rc_loc):
    """Check if an open case can fit before the next resolved case."""
    # Calculate travel time to resolved case
    duration = timedelta(hours=inspection_time)
    case_end_time = current_time + duration
    
    # Add travel time to reach the resolved case
    if next_rc_loc:
        travel_time_min = calculate_travel_time_km(coord, next_rc_loc)
        travel_time = timedelta(minutes=travel_time_min)
        time_needed = case_end_time + travel_time
    else:
        time_needed = case_end_time
    
    # Check if we have enough time before the next resolved case
    can_fit = time_needed <= next_rc_start if next_rc_start else True
    return can_fit, case_end_time

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
    Integrates resolved cases with fixed timeslots and open cases that need scheduling.
    
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
    
    # Log each open case ID to verify they're all present
    if route_with_postal:
        st.session_state.logs.append("Open cases being scheduled:")
        for i, (coord, postal) in enumerate(route_with_postal):
            st.session_state.logs.append(f"  {i+1}. {postal} at {coord}")
    
    if blocked_slots is None:
        blocked_slots = []
        
    # Sort resolved cases by start time
    sorted_resolved_cases = sorted(resolved_cases, key=lambda x: x[0])
    st.session_state.logs.append(f"Sorted resolved cases: {sorted_resolved_cases}")
    
    # Extract the dates of resolved cases to organize scheduling by day
    resolved_dates = {rc[0].date() for rc in sorted_resolved_cases}
    st.session_state.logs.append(f"Resolved case dates: {resolved_dates}")
    
    # Prepare timeslots for all cases (both resolved and open)
    open_case_slots = []
    
    # Group resolved cases by day and timeslot
    resolved_by_day = {}
    for start_dt, end_dt, loc in sorted_resolved_cases:
        day = start_dt.date()
        if day not in resolved_by_day:
            resolved_by_day[day] = []
        resolved_by_day[day].append((start_dt, end_dt, loc))
    
    # Add resolved cases directly to timeslots (they have fixed slots)
    resolved_case_slots = []
    resolved_case_ids = {}  # Track case IDs for each location to maintain consistency

    for idx, (start_dt, end_dt, loc) in enumerate(sorted_resolved_cases):
        # Use a consistent ID for each resolved case based on its location
        loc_key = f"{loc[0]:.6f},{loc[1]:.6f}"
        if loc_key not in resolved_case_ids:
            resolved_case_ids[loc_key] = f"RC{idx+1:03}"
        
        case_id = resolved_case_ids[loc_key]
        st.session_state.logs.append(f"Processing resolved case: {case_id} at {loc} for time {start_dt} to {end_dt}")
        
        # Find if this case is already part of a group
        already_grouped = False
        for group, group_start, group_end_str in resolved_case_slots:
            # Check if this resolved case fits in the time window of an existing group
            group_end = datetime.combine(group_start.date(), 
                                         datetime.strptime(group_end_str, "%H:%M").time())
            
            if (start_dt == group_start and 
                end_dt.time() == datetime.strptime(group_end_str, "%H:%M").time()):
                # Same timeslot, check distance
                can_group = True
                for g_loc, _, _ in group:
                    if geodesic(g_loc, loc).km >= max_distance_km:
                        can_group = False
                        break
                
                if can_group and len(group) < max_cases_per_slot:
                    # Add to existing group
                    group.append((loc, case_id, 1))  # Assume 1 hour for resolved cases
                    already_grouped = True
                    st.session_state.logs.append(f"Added {case_id} to existing group")
                    break
        
        if not already_grouped:
            # Create new group for this resolved case
            case_duration = (end_dt - start_dt).total_seconds() / 3600  # Duration in hours
            resolved_case_slots.append([[(loc, case_id, case_duration)], 
                                       start_dt, 
                                       end_dt.strftime("%H:%M")])
            st.session_state.logs.append(f"Created new group for {case_id}")
    
    # Get locations of all resolved cases for overlap checking
    resolved_locations = {}
    for start_dt, end_dt, loc in sorted_resolved_cases:
        loc_key = f"{loc[0]:.6f},{loc[1]:.6f}"
        if loc_key not in resolved_locations:
            resolved_locations[loc_key] = []
        resolved_locations[loc_key].append((start_dt, end_dt))

    # Filter out open cases that have the same location as resolved cases
    filtered_open_cases = []
    filtered_inspection_times = []
    excluded_indices = []

    # First identify which indices will be excluded
    for i, ((lat, lng), postal) in enumerate(route_with_postal):
        loc_key = f"{lat:.6f},{lng:.6f}"
        # Check if this location closely matches any resolved case location
        is_duplicate = False
        
        # Exact key match
        if loc_key in resolved_locations:
            is_duplicate = True
        else:
            # Try an approximate match if exact match fails
            for resolved_key in resolved_locations:
                resolved_lat, resolved_lng = map(float, resolved_key.split(','))
                # If coordinates are extremely close (within a few meters)
                if abs(lat - resolved_lat) < 0.0001 and abs(lng - resolved_lng) < 0.0001:
                    is_duplicate = True
                    st.session_state.logs.append(f"Approximate match found: Open case {postal} at {lat},{lng} matches resolved case at {resolved_lat},{resolved_lng}")
                    break
        
        if is_duplicate:
            st.session_state.logs.append(f"Excluding open case {postal} at {lat},{lng} because it overlaps with a resolved case")
            excluded_indices.append(i)
    
    # Now create filtered lists that maintain the proper ordering
    for i, ((lat, lng), postal) in enumerate(route_with_postal):
        if i not in excluded_indices:
            filtered_open_cases.append(((lat, lng), postal))
            if i < len(inspection_times):
                filtered_inspection_times.append(inspection_times[i])
            else:
                # Use default if somehow inspection time is missing
                filtered_inspection_times.append(1)

    # Update with filtered cases
    st.session_state.logs.append(f"Original cases: {len(route_with_postal)}, Excluded: {len(excluded_indices)}, Remaining: {len(filtered_open_cases)}")
    route_with_postal = filtered_open_cases
    inspection_times = filtered_inspection_times
    st.session_state.logs.append(f"After filtering overlapping cases: {len(route_with_postal)} open cases remaining")
    
    # Find the next earliest resolved case (if any)
    earliest_rc = None if not sorted_resolved_cases else sorted_resolved_cases[0]
    
    # Schedule open cases
    # Start with the earliest working day after current date at 9 AM
    start_date = add_working_days(current_date, 2, holidays_set)
    base_time = datetime.combine(start_date, time(start_hour, 0))
    current_time = base_time
    
    st.session_state.logs.append(f"Starting scheduling from beginning of working day: {current_time}")
    
    # We'll note the earliest resolved case but won't change our start time to it
    if earliest_rc:
        st.session_state.logs.append(f"Earliest resolved case is at: {earliest_rc[0]}")
    
    # Initialize variables
    group = []
    group_time = timedelta(hours=1)
    prev_group_end_time = None
    prev_group_last_coord = route_with_postal[0][0] if route_with_postal else None

    # Create data structures for working with resolved cases
    sorted_resolved_by_time = sorted(sorted_resolved_cases, key=lambda x: x[0])
    next_resolved_case_idx = 0
    
    def get_next_resolved_case(current_time):
        """Find the next resolved case that's coming up after current_time"""
        for i in range(next_resolved_case_idx, len(sorted_resolved_by_time)):
            rc_start, rc_end, rc_loc = sorted_resolved_by_time[i]
            if rc_start > current_time:
                return i, (rc_start, rc_end, rc_loc)
        return None, None
    
    def can_reach_resolved_case(current_loc, current_time, resolved_start, resolved_loc):
        """Check if we can reach a resolved case from current location in time"""
        travel_time_min = calculate_travel_time_km(current_loc, resolved_loc)
        travel_time = timedelta(minutes=math.ceil(travel_time_min))
        expected_arrival = current_time + travel_time
        buffer_time = timedelta(minutes=15)  # 15 minute buffer
        
        return expected_arrival + buffer_time <= resolved_start, travel_time
    
    def find_nearby_resolved_cases(current_loc, current_time, max_distance=1.0):
        """Find resolved cases that are nearby (within max_distance km) and can be grouped"""
        nearby_cases = []
        
        for i, (rc_start, rc_end, rc_loc) in enumerate(sorted_resolved_by_time):
            if rc_start > current_time and rc_start <= current_time + timedelta(hours=2):
                distance = geodesic(current_loc, rc_loc).km
                
                if distance < max_distance:
                    # Check if we can reasonably reach this case in time
                    can_reach, travel_time = can_reach_resolved_case(current_loc, current_time, rc_start, rc_loc)
                    
                    if can_reach:
                        nearby_cases.append((i, (rc_start, rc_end, rc_loc), distance))
        
        return sorted(nearby_cases, key=lambda x: x[1][0])  # Sort by start time
    
    i = 0
    while i < len(route_with_postal):
        coord, postal = route_with_postal[i]
        inspection_time = inspection_times[i]
        case = (coord, postal, inspection_time)

        # CHANGE 1: If approaching a resolved case, check if we can reach it in time
        next_rc_idx, next_rc = get_next_resolved_case(current_time)
        
        if next_rc:
            rc_start, rc_end, rc_loc = next_rc
            
            # First, calculate expected end time for current case if scheduled at current_time
            expected_end_time = current_time + timedelta(hours=inspection_time)
            
            # Calculate travel time to reach the resolved case
            travel_time_min = calculate_travel_time_km(coord, rc_loc)
            travel_time = timedelta(minutes=math.ceil(travel_time_min))
            
            # Check if we can reach the resolved case on time
            # Need to finish our case, travel, and have a 15-minute buffer
            can_reach = expected_end_time + travel_time + timedelta(minutes=15) <= rc_start
            
            st.session_state.logs.append(f"Case would end at {expected_end_time}, needs {travel_time} to reach RC at {rc_start}, can reach: {can_reach}")
            
            if can_reach and inspection_time <= 1:
                st.session_state.logs.append(f"Can reach resolved case at {rc_start} from current location with {travel_time} travel time")
                
                # CHANGE 3: Check if this open case is near a resolved case and can be grouped
                distance_to_rc = geodesic(coord, rc_loc).km
                
                if distance_to_rc < max_distance_km:
                    st.session_state.logs.append(f"Open case at {coord} is near resolved case at {rc_loc} ({distance_to_rc:.2f} km)")
                    
                    # Find the resolved case slot
                    rc_slot = None
                    for slot_idx, (slot_group, slot_start, slot_end_str) in enumerate(resolved_case_slots):
                        if slot_start == rc_start:
                            rc_slot = (slot_idx, slot_group, slot_start, slot_end_str)
                            break
                    
                    if rc_slot and len(rc_slot[1]) < max_cases_per_slot:
                        slot_idx, slot_group, slot_start, slot_end_str = rc_slot
                        
                        # Add this open case to the resolved case group
                        resolved_case_slots[slot_idx][0].append((coord, postal, inspection_time))
                        st.session_state.logs.append(f"Added open case {postal} to resolved case group at {slot_start}")
                        
                        # Move to the next open case
                        i += 1
                        continue
            else:
                st.session_state.logs.append(f"Cannot reach resolved case at {rc_start} from current location")
                
                # IMPORTANT: Schedule this case after the resolved case instead
                if expected_end_time + travel_time > rc_start:
                    st.session_state.logs.append(f"Not enough time before resolved case, scheduling after it instead")
                    # Schedule current case after the resolved case ends with travel time
                    travel_time_from_rc = timedelta(minutes=math.ceil(travel_time_min / 30) * 30)
                    new_start_time = rc_end + travel_time_from_rc
                    current_time = find_next_available_time(new_start_time, timedelta(hours=inspection_time), blocked_slots, end_hour)
                    if exceeds_end_hour(current_time, timedelta(hours=inspection_time)):
                        current_time = move_to_next_day(current_time)
                        current_time = find_next_available_time(current_time, timedelta(hours=inspection_time), blocked_slots, end_hour)
                
                # If we still have other open cases, consider rerouting them
                if len(route_with_postal) - i > 1:  # If there are more cases to process
                    remaining_coords = [loc for loc, _ in route_with_postal[i:]]
                    
                    # Reroute from the resolved case location
                    try:
                        st.session_state.logs.append(f"Rerouting {len(remaining_coords)} remaining open cases from resolved case at {rc_loc}")
                        new_route = find_shortest_path(remaining_coords, rc_loc)
                        
                        # Map the new route back to cases and inspection times
                        new_cases = []
                        new_times = []
                        
                        for new_coord in new_route:
                            for j in range(i, len(route_with_postal)):
                                original_coord, original_postal = route_with_postal[j]
                                if (abs(original_coord[0] - new_coord[0]) < 0.0001 and 
                                    abs(original_coord[1] - new_coord[1]) < 0.0001):
                                    # Preserve the original postal/case ID when rerouting
                                    new_cases.append((original_coord, original_postal))
                                    new_times.append(inspection_times[j])
                                    break
                        
                        if len(new_cases) == len(route_with_postal) - i:
                            # Successfully remapped all cases
                            route_with_postal = route_with_postal[:i] + new_cases
                            inspection_times = inspection_times[:i] + new_times
                            st.session_state.logs.append(f"Successfully rerouted remaining cases from resolved case")
                        else:
                            st.session_state.logs.append(f"Warning: Could not remap all cases. Found {len(new_cases)} of {len(route_with_postal) - i}")
                    except Exception as e:
                        st.session_state.logs.append(f"Error during rerouting: {str(e)}")
        
        # Process the current case
        if inspection_time > 1:
            # If we have a group in progress, finalize it first
            if group:
                current_time = find_next_available_time(current_time, group_time, blocked_slots, end_hour)
                if exceeds_end_hour(current_time, group_time):
                    current_time = move_to_next_day(current_time)
                    current_time = find_next_available_time(current_time, group_time, blocked_slots, end_hour)
                
                group_end = current_time + group_time
                st.session_state.logs.append(f"Assigning group timeslot: Start {current_time}, End {group_end}")
                open_case_slots.append((group, current_time, group_end.strftime("%H:%M")))
                prev_group_end_time = group_end
                prev_group_last_coord = group[-1][0] if group else prev_group_last_coord
                current_time = group_end
                group = []

            # Calculate travel time from previous case
            if prev_group_end_time is not None:
                travel_time_min = calculate_travel_time_km(prev_group_last_coord, coord)
                rounded_travel_time = timedelta(minutes=math.ceil(travel_time_min / 30) * 30)
                new_start_time = prev_group_end_time + rounded_travel_time
                current_time = find_next_available_time(new_start_time, timedelta(hours=inspection_time), blocked_slots, end_hour)
                if exceeds_end_hour(current_time, timedelta(hours=inspection_time)):
                    current_time = move_to_next_day(current_time)
                    current_time = find_next_available_time(current_time, timedelta(hours=inspection_time), blocked_slots, end_hour)

            # CHANGE 2: Check for conflicts with resolved cases and ensure enough travel time
            for rc_start, rc_end, rc_loc in sorted_resolved_by_time:
                case_end_time = current_time + timedelta(hours=inspection_time)
                
                # First, check if we need to finish an open case before a resolved case starts
                if current_time < rc_start and case_end_time > rc_start:
                    # Calculate travel time needed to go from open case to resolved case
                    travel_time_min = calculate_travel_time_km(coord, rc_loc)
                    travel_time = timedelta(minutes=math.ceil(travel_time_min))
                    
                    # Check if we would have enough time to finish and travel to resolved case
                    if case_end_time + travel_time > rc_start:
                        st.session_state.logs.append(f"Not enough time to finish case and travel to resolved case at {rc_start}")
                        # If not enough time, schedule after the resolved case instead
                        travel_time_after = timedelta(minutes=math.ceil(travel_time_min / 30) * 30)
                        new_start_time = rc_end + travel_time_after
                        current_time = find_next_available_time(new_start_time, timedelta(hours=inspection_time), blocked_slots, end_hour)
                        if exceeds_end_hour(current_time, timedelta(hours=inspection_time)):
                            current_time = move_to_next_day(current_time)
                            current_time = find_next_available_time(current_time, timedelta(hours=inspection_time), blocked_slots, end_hour)
                        break
                
                # Check for direct time overlap
                if (current_time < rc_end and case_end_time > rc_start):
                    st.session_state.logs.append(f"Conflict with resolved case at {rc_start}")
                    # This would overlap, so schedule after the resolved case
                    
                    # Calculate travel time from resolved case to open case
                    travel_time_min = calculate_travel_time_km(rc_loc, coord)
                    travel_time = timedelta(minutes=math.ceil(travel_time_min / 30) * 30)
                    
                    # Ensure enough time to travel from resolved case to open case
                    new_start_time = rc_end + travel_time
                    current_time = find_next_available_time(new_start_time, timedelta(hours=inspection_time), blocked_slots, end_hour)
                    if exceeds_end_hour(current_time, timedelta(hours=inspection_time)):
                        current_time = move_to_next_day(current_time)
                        current_time = find_next_available_time(current_time, timedelta(hours=inspection_time), blocked_slots, end_hour)
                    break

            duration = timedelta(hours=inspection_time)
            current_time = find_next_available_time(current_time, duration, blocked_slots, end_hour)
            
            if exceeds_end_hour(current_time, duration):
                current_time = move_to_next_day(current_time)
                current_time = find_next_available_time(current_time, duration, blocked_slots, end_hour)
            
            start = current_time
            end = current_time + duration
            st.session_state.logs.append(f"Assigning single case timeslot: Start {start}, End {end}")
            open_case_slots.append(([case], start, end.strftime("%H:%M")))
            prev_group_end_time = end
            prev_group_last_coord = coord
            current_time = end
        else:
            # For 1-hour cases, try to group them
            if not group:
                # Starting a new group
                if prev_group_end_time is not None:
                    # CHANGE 2: Calculate travel time from previous location 
                    travel_time_min = calculate_travel_time_km(prev_group_last_coord, coord)
                    rounded_travel_time = timedelta(minutes=math.ceil(travel_time_min / 30) * 30)
                    new_start_time = prev_group_end_time + rounded_travel_time
                    current_time = find_next_available_time(new_start_time, group_time, blocked_slots, end_hour)
                    if exceeds_end_hour(current_time, group_time):
                        current_time = move_to_next_day(current_time)
                        current_time = find_next_available_time(current_time, group_time, blocked_slots, end_hour)
                
                # Check for conflicts with resolved cases
                overlaps_resolved = False
                for rc_start, rc_end, rc_loc in sorted_resolved_by_time:
                    case_end_time = current_time + group_time
                    
                    # Check for direct time overlap
                    if (current_time < rc_end and case_end_time > rc_start):
                        st.session_state.logs.append(f"Conflict with resolved case at {rc_start}")
                        
                        # Calculate travel time from resolved case to open case
                        travel_time_min = calculate_travel_time_km(rc_loc, coord)
                        travel_time = timedelta(minutes=math.ceil(travel_time_min / 30) * 30)
                        
                        # Ensure enough time to travel from resolved case to open case
                        new_start_time = rc_end + travel_time
                        current_time = find_next_available_time(new_start_time, group_time, blocked_slots, end_hour)
                        if exceeds_end_hour(current_time, group_time):
                            current_time = move_to_next_day(current_time)
                            current_time = find_next_available_time(current_time, group_time, blocked_slots, end_hour)
                        overlaps_resolved = True
                        break
                
                if not overlaps_resolved:
                    current_time = find_next_available_time(current_time, group_time, blocked_slots, end_hour)
                    if exceeds_end_hour(current_time, group_time):
                        current_time = move_to_next_day(current_time)
                        current_time = find_next_available_time(current_time, group_time, blocked_slots, end_hour)
                
                group.append(case)
            else:
                # We already have a group in progress, check if we can add this case
                last_coord, _, _ = group[-1]
                distance_km = geodesic(last_coord, coord).km
                st.session_state.logs.append(f"Distance from {last_coord} to {coord}: {distance_km} km")
                
                if distance_km < max_distance_km and len(group) < max_cases_per_slot:
                    # Can add to existing group
                    group.append(case)
                else:
                    # Finalize current group and start a new one
                    current_time = find_next_available_time(current_time, group_time, blocked_slots, end_hour)
                    if exceeds_end_hour(current_time, group_time):
                        current_time = move_to_next_day(current_time)
                        current_time = find_next_available_time(current_time, group_time, blocked_slots, end_hour)
                    
                    group_end = current_time + group_time
                    st.session_state.logs.append(f"Assigning group timeslot: Start {current_time}, End {group_end}")
                    open_case_slots.append((group, current_time, group_end.strftime("%H:%M")))
                    prev_group_end_time = group_end
                    prev_group_last_coord = group[-1][0]
                    current_time = group_end
                    
                    # Calculate travel time to the next case
                    travel_time_min = calculate_travel_time_km(prev_group_last_coord, coord)
                    rounded_travel_time = timedelta(minutes=math.ceil(travel_time_min / 30) * 30)
                    new_start_time = prev_group_end_time + rounded_travel_time
                    current_time = find_next_available_time(new_start_time, group_time, blocked_slots, end_hour)
                    if exceeds_end_hour(current_time, group_time):
                        current_time = move_to_next_day(current_time)
                        current_time = find_next_available_time(current_time, group_time, blocked_slots, end_hour)
                    
                    # Start new group with this case
                    group = [case]

        # Move to next case
        i += 1

    # Handle any remaining group
    if group:
        current_time = find_next_available_time(current_time, group_time, blocked_slots, end_hour)
        if exceeds_end_hour(current_time, group_time):
            current_time = move_to_next_day(current_time)
            current_time = find_next_available_time(current_time, group_time, blocked_slots, end_hour)
        
        group_end = current_time + group_time
        st.session_state.logs.append(f"Assigning final group timeslot: Start {current_time}, End {group_end}")
        open_case_slots.append((group, current_time, group_end.strftime("%H:%M")))
    
    # Combine all timeslots (resolved and open cases)
    all_slots = resolved_case_slots + open_case_slots
    
    # Sort all slots by start time
    all_slots.sort(key=lambda x: x[1])
    
    # Format timeslots for display
    result_timeslots = []
    for group, start_time, end_time_str in all_slots:
        start_time_str = start_time.strftime("%d/%m, %H:%M")
        time_slot = f"({start_time_str} - {end_time_str})"
        for coord, postal, _ in group:
            # Ensure we're using the correct case ID for this coordinate if it's an open case
            if not postal.startswith("RC"):
                coord_key = f"{coord[0]:.6f},{coord[1]:.6f}"
                # Check if we have a stored ID for this coordinate
                if hasattr(st.session_state, 'open_case_ids'):
                    for stored_key, stored_id in st.session_state.open_case_ids.items():
                        try:
                            stored_lat, stored_lng = map(float, stored_key.split(','))
                            if abs(coord[0] - stored_lat) < 0.0001 and abs(coord[1] - stored_lng) < 0.0001:
                                postal = stored_id
                                break
                        except Exception as e:
                            st.session_state.logs.append(f"Error matching case ID: {e}")
            
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
# Add case IDs for open cases to maintain consistency
if "open_case_ids" not in st.session_state:
    st.session_state.open_case_ids = {}  # Maps coordinate keys to case IDs
if "open_case_counter" not in st.session_state:
    st.session_state.open_case_counter = 0



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
    st.sidebar.text_area("Resolved Cases Information", resolved_cases_info, height=200)
else:
    st.sidebar.text("No resolved cases yet")

# Add Open Cases section to the sidebar
st.sidebar.subheader("Open Cases")
if st.session_state.open_cases:
    open_cases_info = ""
    for i, point in enumerate(st.session_state.open_cases):
        lat, lng = point
        coord_key = f"{lat:.6f},{lng:.6f}"
        # Use the stored case ID instead of generating a new one
        case_id = st.session_state.open_case_ids.get(coord_key, f"OC{i+1:03}")
        
        # Use coordinate key for inspection time lookup
        inspection_time = st.session_state.inspection_times.get(coord_key, 1)
        
        # Get the timeslot if available in route_data
        timeslot = "Not scheduled yet"
        if st.session_state.route_data:
            for coord, postal, ts in st.session_state.route_data:
                if (abs(coord[0] - lat) < 0.0001 and abs(coord[1] - lng) < 0.0001):
                    # Check if the postal code matches our case_id or starts with OC
                    if postal == case_id or postal.startswith("OC"):
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

if blocked_times:
    st.session_state.blocked_slots = parse_blocked_times(blocked_times)
if holidays:
    st.session_state.holiday_set = parse_holidays(holidays_str)


# Create the base map
if st.session_state.route_data or st.session_state.resolved_cases:
    # Show the route map if we have calculated a route or have resolved cases
    map_center = None
    
    if st.session_state.route_data and len(st.session_state.route_data) > 0:
        map_center = st.session_state.route_data[0][0]
        route_with_timeslot = st.session_state.route_data
    else:
        route_with_timeslot = []
        
    if not map_center and st.session_state.resolved_cases and len(st.session_state.resolved_cases) > 0:
        map_center = st.session_state.resolved_cases[0]["location"]
    
    if not map_center:
        map_center = [1.3521, 103.8198]  # Default to Singapore center
    
    m = folium.Map(location=map_center, zoom_start=13)
    
    # Draw the scheduled route (ordered by timeslot)
    for i, (coord, postal, timeslot) in enumerate(route_with_timeslot):
        # Determine if this is an open case or resolved case based on the postal code format
        is_resolved = postal.startswith("RC")
        # Use different colors for different types of points
        color = "green" if is_resolved else "blue"
        icon_type = "info-sign" if is_resolved else "info"
        
        # For open cases, ensure we're displaying the correct case ID from our stored mapping
        if not is_resolved:
            coord_key = f"{coord[0]:.6f},{coord[1]:.6f}"
            # Double check if this coord matches a stored case ID
            if hasattr(st.session_state, 'open_case_ids'):
                for stored_key, stored_id in st.session_state.open_case_ids.items():
                    try:
                        stored_lat, stored_lng = map(float, stored_key.split(','))
                        if abs(coord[0] - stored_lat) < 0.0001 and abs(coord[1] - stored_lng) < 0.0001:
                            postal = stored_id
                            break
                    except Exception as e:
                        st.session_state.logs.append(f"Error matching case ID: {e}")
        
        folium.Marker(
            coord, 
            icon=folium.Icon(color=color, icon=icon_type),
            tooltip=f"{postal}: {timeslot}"
        ).add_to(m)
        
        # Add text label
        folium.map.Marker(
            location=coord,
            icon=folium.DivIcon(
                icon_size=(150, 36),
                html=f'<div style="font-size: 12px; color: black; background-color: white; padding: 2px; border-radius: 4px; border: 1px solid grey;">{postal}<br>{timeslot}</div>'
            )
        ).add_to(m)
    
    # Additionally, ensure all resolved cases are shown even if they're not in the route
    for case in st.session_state.resolved_cases:
        loc = case["location"]
        timeslot = case["timeslot"]
        case_id = case["case_id"]
        
        # Check if this resolved case is already in the route
        already_in_route = False
        for c, p, _ in route_with_timeslot:
            # Check exact coordinates match and case ID format
            if (abs(c[0] - loc[0]) < 0.0001 and abs(c[1] - loc[1]) < 0.0001):
                # This location is already in the route
                already_in_route = True
                # Log to debug
                st.session_state.logs.append(f"Resolved case {case_id} at {loc} is already in route with ID {p}")
                break
        
        # Only add markers that are not already in the route
        if not already_in_route:
            st.session_state.logs.append(f"Adding resolved case {case_id} at {loc} to map (not in route)")
            folium.Marker(
                loc, 
                icon=folium.Icon(color="green", icon="info-sign"),
                tooltip=f"{case_id}: {timeslot}"
            ).add_to(m)
            
            folium.map.Marker(
                location=loc,
                icon=folium.DivIcon(
                    icon_size=(150, 36),
                    html=f'<div style="font-size: 12px; color: black; background-color: white; padding: 2px; border-radius: 4px; border: 1px solid grey;">{case_id}<br>{timeslot}</div>'
                )
            ).add_to(m)
    
    # Draw lines connecting points in order of timeslot (earliest to latest)
    if route_with_timeslot:
        for i in range(len(route_with_timeslot) - 1):
            c1, p1, t1 = route_with_timeslot[i]
            c2, p2, t2 = route_with_timeslot[i + 1]
            dist_km = geodesic(c1, c2).km
            midpoint = ((c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2)
            
            # Skip drawing distance label if distance is extremely small (same location)
            if dist_km < 0.01:
                st.session_state.logs.append(f"Skipping distance label for {p1} to {p2} (distance too small: {dist_km:.6f} km)")
                continue
            
            # Use different line colors based on the type of connection
            p1_resolved = p1.startswith("RC")
            p2_resolved = p2.startswith("RC")
            
            if p1_resolved and p2_resolved:
                line_color = "green"  # Both are resolved cases
            elif not p1_resolved and not p2_resolved:
                line_color = "blue"   # Both are open cases
            else:
                line_color = "orange" # Mixed connection
            
            folium.PolyLine([c1, c2], color=line_color, weight=3, opacity=0.8).add_to(m)
            
            # Add distance label
            folium.Marker(
                midpoint, 
                icon=folium.DivIcon(
            html=f'<div style="font-size: 11px; color: black;">{dist_km:.2f} km</div>'
                )
            ).add_to(m)
    
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


# Display the map and handle drawn open_cases
output = st_folium(m, width=700, height=500, key="map", returned_objects=["last_active_drawing", "all_drawings"])

# Process drawn cases depending on the current mode
if output["last_active_drawing"]:
    coords = output["last_active_drawing"]["geometry"]["coordinates"]
    latlng = (coords[1], coords[0])
    if st.session_state.add_mode == "open":
        if latlng not in st.session_state.open_cases:
            st.session_state.open_cases.append(latlng)
            # Assign a consistent ID to this new open case
            st.session_state.open_case_counter += 1
            coord_key = f"{latlng[0]:.6f},{latlng[1]:.6f}"
            st.session_state.open_case_ids[coord_key] = f"OC{st.session_state.open_case_counter:03}"
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
            value="11/04/2025 11:00 - 12:00",  # Default value
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
    # We don't need to reindex open_cases anymore since the route is now ordered by timeslot
    # And contains both open and resolved cases
    # The route display logic handles the ordering based on timeslots


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
        if len(st.session_state.open_cases) < 1 and len(st.session_state.resolved_cases) < 1:
            st.warning("Add at least 1 open case or 1 resolved case.")
        else:
            # Determine initial starting point - it could be either an open case 
            # or the earliest resolved case depending on the timings
            startCoords = None
            is_start_resolved = False
            
            # If we have resolved cases, check to see if any of them should be the starting point
            earliest_resolved_time = None
            earliest_resolved_loc = None
            
            # Track resolved case IDs
            resolved_case_locs = {}
            for case in st.session_state.resolved_cases:
                try:
                    slot_str = case["timeslot"]
                    loc = case["location"]
                    case_id = case["case_id"]
                    
                    # Store the case ID for this location
                    loc_key = f"{loc[0]:.6f},{loc[1]:.6f}"
                    resolved_case_locs[loc_key] = case_id
                    
                    # Parse the datetime from the slot string
                    date_part, times_part = slot_str.split(" ", 1)
                    start_time_str = times_part.split(" - ")[0]
                    
                    slot_date = datetime.strptime(date_part, "%d/%m/%Y").date()
                    start_dt = datetime.combine(slot_date, datetime.strptime(start_time_str, "%H:%M").time())
                    
                    if earliest_resolved_time is None or start_dt < earliest_resolved_time:
                        earliest_resolved_time = start_dt
                        earliest_resolved_loc = loc
                except ValueError as e:
                    st.session_state.logs.append(f"Error parsing resolved slot '{case['timeslot']}': {e}")
            
            # If we have open cases, use the first one as a potential starting point
            if st.session_state.open_cases:
                default_start = st.session_state.open_cases[0]
            else:
                # If no open cases, use the earliest resolved case
                default_start = earliest_resolved_loc if earliest_resolved_loc else [1.3521, 103.8198]  # Default to Singapore
            
            # Choose the best starting point
            if earliest_resolved_time and earliest_resolved_loc:
                # If the earliest resolved case is within 2 days, start from there
                start_date = add_working_days(st.session_state.current_date, 2, st.session_state.holiday_set)
                start_datetime = datetime.combine(start_date, time(9, 0))
                
                if earliest_resolved_time <= start_datetime + timedelta(days=1):
                    startCoords = earliest_resolved_loc
                    is_start_resolved = True
                    st.session_state.logs.append(f"Starting route from earliest resolved case at {earliest_resolved_time}")
                else:
                    startCoords = default_start
                    st.session_state.logs.append(f"Starting route from default point {default_start}")
            else:
                startCoords = default_start
                st.session_state.logs.append(f"No resolved cases, starting from {default_start}")
            
            # Calculate route for open cases
            if st.session_state.open_cases:
                # Log the original open cases
                st.session_state.logs.append(f"Original open cases: {st.session_state.open_cases}")
                
                # Find the shortest path first
                shortest_route = find_shortest_path(st.session_state.open_cases, startCoords)
                st.session_state.logs.append(f"Shortest route: {shortest_route}")
                
                # Create route with proper case IDs
                route_with_postal = []
                for i, coord in enumerate(shortest_route):
                    # Skip the starting point if it's a resolved case and not in the open_cases
                    if is_start_resolved and i == 0 and coord == startCoords and coord not in st.session_state.open_cases:
                        st.session_state.logs.append(f"Skipping starting point (resolved case) in open case numbering")
                        continue
                        
                    # Use stored case ID if available, otherwise create a new one
                    coord_key = f"{coord[0]:.6f},{coord[1]:.6f}"
                    if coord_key in st.session_state.open_case_ids:
                        case_id = st.session_state.open_case_ids[coord_key]
                    else:
                        # Only generate a new ID if we don't have one yet
                        st.session_state.open_case_counter += 1
                        case_id = f"OC{st.session_state.open_case_counter:03}"
                        st.session_state.open_case_ids[coord_key] = case_id
                    
                    route_with_postal.append((coord, case_id))
            
                # Get inspection time inputs (ensure they retain the previous values)
                inspection_times = []
                for i, coord in enumerate(shortest_route):
                    # Skip the starting point if it's a resolved case and not in the open_cases
                    if is_start_resolved and i == 0 and coord == startCoords and coord not in st.session_state.open_cases:
                        st.session_state.logs.append(f"Skipping starting point (resolved case) in open case numbering")
                        continue
                    
                    # Use coordinate as key for inspection time lookup
                    coord_key = f"{coord[0]:.6f},{coord[1]:.6f}"
                    inspection_time = st.session_state.inspection_times.get(coord_key, 1)  # Default to 1 if not set
                    st.session_state.logs.append(f"Using inspection time {inspection_time} for coordinate {coord}")
                    inspection_times.append(inspection_time)
                else:
                    # If there are no open cases, create an empty route
                    route_with_postal = []
                    inspection_times = []
            
            # Log for debugging
            st.session_state.logs.append(f"Route with postal: {route_with_postal}")
            st.session_state.logs.append(f"Inspection times: {inspection_times}")
            st.session_state.logs.append(f"Parsing {len(parsed_resolved_cases)} resolved cases")
            
            # Assign timeslots considering both open and resolved cases
            route_with_timeslot = assign_timeslots_stable_with_travel_time(
                route_with_postal=route_with_postal,
                inspection_times=inspection_times,
                resolved_cases=parsed_resolved_cases,
                blocked_slots=st.session_state.blocked_slots,
                holidays_set=st.session_state.holiday_set,
                current_date=st.session_state.current_date
            )
            
            # Log the output for debugging
            st.session_state.logs.append(f"Route with timeslot (before sorting): {route_with_timeslot}")
            
            # Reorder route by timeslot for display
            # Extract datetime from timeslot string to sort
            def extract_datetime(timeslot_str):
                # Format: "(dd/mm, HH:MM - HH:MM)"
                date_time_part = timeslot_str.strip("()").split(" - ")[0]
                return datetime.strptime(date_time_part, "%d/%m, %H:%M")
            
            # Sort by timeslot
            route_with_timeslot.sort(key=lambda x: extract_datetime(x[2]))
            
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
    st.session_state.logs.append("Current open_cases:")
    for i, point in enumerate(st.session_state.open_cases):
        lat, lng = point
        # Get the stored case ID for this point
        coord_key = f"{lat:.6f},{lng:.6f}"
        case_id = st.session_state.open_case_ids.get(coord_key, f"OC{i+1:03}")
        st.session_state.logs.append(f"Point {case_id}: Latitude {lat:.4f}, Longitude {lng:.4f}")
        
        # Use coordinate as key for inspection time
        
        # Add an inspection time input for each point
        inspection_time = st.number_input(
            f"Inspection Time for {case_id}", 
            min_value=1, 
            max_value=8, 
            value=st.session_state.inspection_times.get(coord_key, 1), 
            key=f"inspection_time_{i}"
        )
        st.session_state.inspection_times[coord_key] = inspection_time  # Store with coordinate key
        
        # Add a delete button for each point
        delete_button = st.button(f"❌ Delete {case_id}", key=f"delete_{i}")
        if delete_button:
            # Also remove the inspection time when deleting the point
            if coord_key in st.session_state.inspection_times:
                del st.session_state.inspection_times[coord_key]
            # Also remove the case ID when deleting the point
            if coord_key in st.session_state.open_case_ids:
                del st.session_state.open_case_ids[coord_key]
            
            st.session_state.open_cases.pop(i)
            st.session_state.show_route = False  # Reset the route when a point is deleted
            st.session_state.route_data = None  # Clear the route data
            st.success(f"✅ Case {case_id} deleted successfully.")

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
#endregion