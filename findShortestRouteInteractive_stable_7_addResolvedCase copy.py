import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from geopy.distance import geodesic
from datetime import datetime, timedelta, time as datetime_time
import math
# Add this at the top with other imports
from streamlit.components.v1 import html

# Ensure we don't have a time variable that could shadow the datetime.time class
if 'time' in locals() or 'time' in globals():
    del time

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
    Now supports both open cases and resolved cases with fixed timeslots.
    
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
    
    # Combine blocked slots with resolved_cases times to ensure no conflicts
    combined_blocked_slots = blocked_slots.copy()
    resolved_locations = {}
    
    # Sort resolved cases chronologically
    sorted_resolved_cases = sorted(resolved_cases, key=lambda x: x[0])
    
    for start_dt, end_dt, location in sorted_resolved_cases:
        combined_blocked_slots.append((start_dt, end_dt))
        # Store the location for each resolved case timeslot for travel time calculations
        resolved_locations[(start_dt, end_dt)] = location
    
    # If no open cases, just return an empty list
    if not route_with_postal:
        return []
    
    # If no resolved cases, use the existing logic
    if not resolved_cases:
        timeslots = []
        start_date = add_working_days(current_date, 2, holidays_set)
        base_time = datetime.combine(start_date, datetime_time(start_hour, 0))
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
                    current_time = find_next_available_time(current_time, group_time, combined_blocked_slots, end_hour)
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
                    current_time = find_next_available_time(new_start_time, timedelta(hours=inspection_time), combined_blocked_slots, end_hour)
                    if exceeds_end_hour(current_time, timedelta()):
                        current_time = move_to_next_day(current_time)
                
                duration = timedelta(hours=inspection_time)
                if exceeds_end_hour(current_time, duration):
                    current_time = move_to_next_day(current_time)
                
                current_time = find_next_available_time(current_time, duration, combined_blocked_slots, end_hour)
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
                        current_time = find_next_available_time(new_start_time, group_time, combined_blocked_slots, end_hour)
                        if exceeds_end_hour(current_time, timedelta()):
                            current_time = move_to_next_day(current_time)
                    
                    current_time = find_next_available_time(current_time, group_time, combined_blocked_slots, end_hour)
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
                        current_time = find_next_available_time(current_time, group_time, combined_blocked_slots, end_hour)
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
            current_time = find_next_available_time(current_time, group_time, combined_blocked_slots, end_hour)
            if exceeds_end_hour(current_time, group_time):
                current_time = move_to_next_day(current_time)
            group_end = current_time + group_time
            st.session_state.logs.append(f"Assigning final group timeslot: Start {current_time}, End {group_end}")
            timeslots.append((group, current_time, group_end.strftime("%H:%M")))
    else:
        # Enhanced logic for handling both open and resolved cases
        timeslots = []
        start_date = add_working_days(current_date, 2, holidays_set)
        st.session_state.logs.append(f"Debug: start_date={start_date}, start_hour={start_hour}, type(start_hour)={type(start_hour)}")
        st.session_state.logs.append(f"Debug: datetime_time={datetime_time}, type(datetime_time)={type(datetime_time)}")
        
        try:
            base_time = datetime.combine(start_date, datetime_time(start_hour, 0))
            st.session_state.logs.append(f"Debug: base_time created successfully: {base_time}")
        except Exception as e:
            st.session_state.logs.append(f"Error creating base_time: {e}")
            # Fallback to direct creation
            base_time = datetime(start_date.year, start_date.month, start_date.day, start_hour, 0)
        
        current_time = base_time
        group = []
        group_time = timedelta(hours=1)
        prev_group_end_time = None
        prev_group_last_coord = None
        
        # Helper functions (same as above, just keeping them inside the else block)
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
        
        # Convert route_with_postal to a list of cases with inspection times
        open_cases = []
        for i, (coord, postal) in enumerate(route_with_postal):
            inspection_time = inspection_times[i]
            open_cases.append((coord, postal, inspection_time))
        
        # Create a timeline of events (both resolved cases and potential open case slots)
        timeline = []
        
        # Add resolved cases to timeline
        for start_dt, end_dt, location in sorted_resolved_cases:
            timeline.append({
                'type': 'resolved',
                'start': start_dt,
                'end': end_dt,
                'location': location,
                'duration': end_dt - start_dt
            })
        
        # Initialize with starting from the first resolved case if available, otherwise base_time
        if timeline:
            current_time = min(timeline[0]['start'], base_time)
            # If the first resolved case is earlier than base_time, adjust to base_time
            if current_time < base_time:
                current_time = base_time
            
            if timeline[0]['start'] > base_time:
                # We can schedule open cases before the first resolved case
                prev_group_last_coord = route_with_postal[0][0]
                prev_group_end_time = base_time
        else:
            current_time = base_time
            prev_group_last_coord = route_with_postal[0][0]
            prev_group_end_time = base_time
        
        # Process open cases and resolved cases in chronological order
        remaining_open_cases = open_cases.copy()
        processed_timeline_indices = set()
        processed_open_case_indices = set()
        
        # Keep processing until all open cases are assigned
        while remaining_open_cases:
            st.session_state.logs.append(f"Current time: {current_time}, Remaining open cases: {len(remaining_open_cases)}")
            
            # Find the next resolved case if any
            next_resolved_idx = None
            for i, event in enumerate(timeline):
                if i not in processed_timeline_indices and event['start'] > current_time:
                    if next_resolved_idx is None or event['start'] < timeline[next_resolved_idx]['start']:
                        next_resolved_idx = i
            
            # If there's a next resolved case, we need to ensure we can handle cases before it
            if next_resolved_idx is not None:
                next_resolved = timeline[next_resolved_idx]
                time_until_next_resolved = next_resolved['start'] - current_time
                st.session_state.logs.append(f"Next resolved case at {next_resolved['start']}, time until: {time_until_next_resolved}")
                
                # Find open cases that can fit before the next resolved case
                feasible_cases = []
                
                # First handle any ongoing group
                if group:
                    # Finalize the current group if we have one
                    available_time = find_next_available_time(current_time, group_time, blocked_slots, end_hour)
                    if available_time + group_time <= next_resolved['start']:
                        # We can fit this group before the next resolved case
                        group_end = available_time + group_time
                        st.session_state.logs.append(f"Assigning group before resolved case: Start {available_time}, End {group_end}")
                        timeslots.append((group, available_time, group_end.strftime("%H:%M")))
                        prev_group_end_time = group_end
                        prev_group_last_coord = group[-1][0]
                        current_time = group_end
                        
                        # Mark these cases as processed
                        for case in group:
                            for i, open_case in enumerate(remaining_open_cases):
                                if case[0] == open_case[0] and case[1] == open_case[1]:
                                    processed_open_case_indices.add(i)
                    else:
                        # Can't fit this group before the resolved case, we'll reconsider these cases
                        st.session_state.logs.append(f"Group can't fit before resolved case, will reconsider later")
                    
                    group = []
                
                # Try to fit individual cases or new groups before the next resolved case
                candidates = []
                for i, case in enumerate(remaining_open_cases):
                    if i in processed_open_case_indices:
                        continue
                    
                    coord, postal, inspection_time = case
                    
                    # Check if we have time to travel to this case and complete it before the next resolved case
                    travel_time = timedelta(minutes=0)
                    if prev_group_last_coord:
                        travel_time_min = calculate_travel_time_km(prev_group_last_coord, coord)
                        travel_time = timedelta(minutes=math.ceil(travel_time_min / 30) * 30)
                    
                    case_duration = timedelta(hours=inspection_time)
                    earliest_start = current_time + travel_time
                    
                    # Find when this case could actually start (considering blocked slots)
                    available_start = find_next_available_time(earliest_start, case_duration, blocked_slots, end_hour)
                    
                    # Check if it can finish before the next resolved case
                    if available_start + case_duration <= next_resolved['start']:
                        # Also check if we'd have enough time to travel to the resolved case afterward
                        travel_to_resolved_min = calculate_travel_time_km(coord, next_resolved['location'])
                        travel_to_resolved = timedelta(minutes=math.ceil(travel_to_resolved_min / 30) * 30)
                        
                        if available_start + case_duration + travel_to_resolved <= next_resolved['start']:
                            candidates.append((i, case, available_start, case_duration))
                
                # Sort candidates by start time
                candidates.sort(key=lambda x: x[2])
                
                # Process candidates
                if candidates:
                    while candidates and group_time_fits_before_resolved(group, candidates[0], next_resolved, max_distance_km, max_cases_per_slot):
                        idx, case, avail_start, duration = candidates.pop(0)
                        coord, postal, inspection_time = case
                        
                        if inspection_time <= 1 and can_group_with_existing(group, coord, max_distance_km, max_cases_per_slot):
                            # Can be grouped with existing short cases
                            if not group:
                                current_time = avail_start
                            group.append(case)
                            processed_open_case_indices.add(idx)
                        else:
                            # Handle as individual case or start a new group
                            if group:
                                # Finalize existing group first
                                group_end = current_time + group_time
                                st.session_state.logs.append(f"Assigning group before individual case: Start {current_time}, End {group_end}")
                                timeslots.append((group, current_time, group_end.strftime("%H:%M")))
                                prev_group_end_time = group_end
                                prev_group_last_coord = group[-1][0]
                                current_time = group_end
                                group = []
                            
                            if inspection_time <= 1:
                                # Start a new group with this short case
                                current_time = avail_start
                                group.append(case)
                                processed_open_case_indices.add(idx)
                            else:
                                # Handle as individual long case
                                end_time = avail_start + duration
                                st.session_state.logs.append(f"Assigning individual case: Start {avail_start}, End {end_time}")
                                timeslots.append(([case], avail_start, end_time.strftime("%H:%M")))
                                prev_group_end_time = end_time
                                prev_group_last_coord = coord
                                current_time = end_time
                                processed_open_case_indices.add(idx)
                
                # Finalize any remaining group
                if group:
                    available_time = find_next_available_time(current_time, group_time, blocked_slots, end_hour)
                    if available_time + group_time <= next_resolved['start']:
                        # We can fit this group before the next resolved case
                        group_end = available_time + group_time
                        st.session_state.logs.append(f"Assigning final group before resolved case: Start {available_time}, End {group_end}")
                        timeslots.append((group, available_time, group_end.strftime("%H:%M")))
                        prev_group_end_time = group_end
                        prev_group_last_coord = group[-1][0]
                        current_time = group_end
                        
                        # Mark these cases as processed
                        for case in group:
                            for i, open_case in enumerate(remaining_open_cases):
                                if case[0] == open_case[0] and case[1] == open_case[1]:
                                    processed_open_case_indices.add(i)
                    group = []
                
                # Now handle the resolved case
                # Set current time to the end of the resolved case
                current_time = next_resolved['end']
                prev_group_end_time = next_resolved['end']
                prev_group_last_coord = next_resolved['location']
                processed_timeline_indices.add(next_resolved_idx)
                
            else:
                # No more resolved cases, process remaining open cases with regular logic
                for i, case in enumerate(remaining_open_cases):
                    if i in processed_open_case_indices:
                        continue
                    
                    coord, postal, inspection_time = case
                    
                    if inspection_time > 1:
                        # Handle long case
                        if group:
                            # Finalize existing group first
                            current_time = find_next_available_time(current_time, group_time, combined_blocked_slots, end_hour)
                            if exceeds_end_hour(current_time, group_time):
                                current_time = move_to_next_day(current_time)
                            group_end = current_time + group_time
                            st.session_state.logs.append(f"Assigning group timeslot: Start {current_time}, End {group_end}")
                            timeslots.append((group, current_time, group_end.strftime("%H:%M")))
                            prev_group_end_time = group_end
                            prev_group_last_coord = group[-1][0] if group else prev_group_last_coord
                            current_time = group_end
                            group = []
                        
                        # Calculate travel time to this case
                        if prev_group_end_time is not None:
                            travel_time_min = calculate_travel_time_km(prev_group_last_coord, coord)
                            rounded_travel_time = timedelta(minutes=math.ceil(travel_time_min / 30) * 30)
                            new_start_time = prev_group_end_time + rounded_travel_time
                            current_time = find_next_available_time(new_start_time, timedelta(hours=inspection_time), combined_blocked_slots, end_hour)
                            if exceeds_end_hour(current_time, timedelta()):
                                current_time = move_to_next_day(current_time)
                        
                        duration = timedelta(hours=inspection_time)
                        if exceeds_end_hour(current_time, duration):
                            current_time = move_to_next_day(current_time)
                        
                        current_time = find_next_available_time(current_time, duration, combined_blocked_slots, end_hour)
                        start = current_time
                        end = current_time + duration
                        st.session_state.logs.append(f"Assigning single case timeslot: Start {start}, End {end}")
                        timeslots.append(([case], start, end.strftime("%H:%M")))
                        prev_group_end_time = end
                        prev_group_last_coord = coord
                        current_time = end
                        processed_open_case_indices.add(i)
                    else:
                        # Handle short case
                        if not group:
                            if prev_group_end_time is not None:
                                travel_time_min = calculate_travel_time_km(prev_group_last_coord, coord)
                                rounded_travel_time = timedelta(minutes=math.ceil(travel_time_min / 30) * 30)
                                new_start_time = prev_group_end_time + rounded_travel_time
                                current_time = find_next_available_time(new_start_time, group_time, combined_blocked_slots, end_hour)
                                if exceeds_end_hour(current_time, timedelta()):
                                    current_time = move_to_next_day(current_time)
                            
                            current_time = find_next_available_time(current_time, group_time, combined_blocked_slots, end_hour)
                            if exceeds_end_hour(current_time, group_time):
                                current_time = move_to_next_day(current_time)
                            group.append(case)
                            processed_open_case_indices.add(i)
                        else:
                            last_coord, _, _ = group[-1]
                            distance_km = geodesic(last_coord, coord).km
                            st.session_state.logs.append(f"Distance from {last_coord} to {coord}: {distance_km} km")
                            if distance_km < max_distance_km and len(group) < max_cases_per_slot:
                                group.append(case)
                                processed_open_case_indices.add(i)
                            else:
                                current_time = find_next_available_time(current_time, group_time, combined_blocked_slots, end_hour)
                                if exceeds_end_hour(current_time, group_time):
                                    current_time = move_to_next_day(current_time)
                                group_end = current_time + group_time
                                st.session_state.logs.append(f"Assigning group timeslot: Start {current_time}, End {group_end}")
                                timeslots.append((group, current_time, group_end.strftime("%H:%M")))
                                prev_group_end_time = group_end
                                prev_group_last_coord = group[-1][0]
                                current_time = group_end
                                group = [case]
                                processed_open_case_indices.add(i)
                                
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
            
            # Remove processed cases from remaining_open_cases
            remaining_open_cases = [case for i, case in enumerate(remaining_open_cases) if i not in processed_open_case_indices]
            processed_open_case_indices = set()
            
            # If we've made no progress this iteration, move to the next day to avoid an infinite loop
            if not remaining_open_cases:
                break
        
        # Handle any remaining group
        if group:
            current_time = find_next_available_time(current_time, group_time, combined_blocked_slots, end_hour)
            if exceeds_end_hour(current_time, group_time):
                current_time = move_to_next_day(current_time)
            group_end = current_time + group_time
            st.session_state.logs.append(f"Assigning final group timeslot: Start {current_time}, End {group_end}")
            timeslots.append((group, current_time, group_end.strftime("%H:%M")))
    
    # Format and return the results
    result_timeslots = []
    
    # Add all resolved cases to the result
    for start_dt, end_dt, location in resolved_cases:
        time_slot = f"({start_dt.strftime('%d/%m, %H:%M')} - {end_dt.strftime('%H:%M')})"
        result_timeslots.append((location, f"RC{len(result_timeslots)+1:03}", time_slot))
    
    # Add all scheduled open cases to the result
    for group, start_time, end_time_str in timeslots:
        start_time_str = start_time.strftime("%d/%m, %H:%M")
        time_slot = f"({start_time_str} - {end_time_str})"
        for coord, postal, _ in group:
            result_timeslots.append((coord, postal, time_slot))
    
    # Sort all timeslots chronologically
    result_timeslots.sort(key=lambda x: parse_timeslot(x[2]))
    
    return result_timeslots

# Helper function to check if a group and new case can fit in a timeslot before a resolved case
def group_time_fits_before_resolved(group, candidate, next_resolved, max_distance_km, max_cases_per_slot):
    if not group:
        return True
    
    idx, case, avail_start, duration = candidate
    coord, postal, inspection_time = case
    
    if inspection_time > 1:
        # Long cases don't go in groups
        return False
    
    if len(group) >= max_cases_per_slot:
        # Already at max group size
        return False
    
    # Check distance constraint
    last_coord = group[-1][0]
    distance_km = geodesic(last_coord, coord).km
    if distance_km >= max_distance_km:
        return False
    
    # Check if there's still time to fit this group before the resolved case
    group_time = timedelta(hours=1)
    if avail_start + group_time > next_resolved['start']:
        return False
    
    # Check if we can reach the resolved case after completing this group
    travel_to_resolved_min = calculate_travel_time_km(coord, next_resolved['location'])
    travel_to_resolved = timedelta(minutes=math.ceil(travel_to_resolved_min / 30) * 30)
    
    return avail_start + group_time + travel_to_resolved <= next_resolved['start']

# Helper function to check if a case can be grouped with existing cases
def can_group_with_existing(group, coord, max_distance_km, max_cases_per_slot):
    if not group or len(group) >= max_cases_per_slot:
        return True  # Empty group or already at max capacity
    
    last_coord = group[-1][0]
    distance_km = geodesic(last_coord, coord).km
    return distance_km < max_distance_km

# Helper function to parse a timeslot string into a datetime object (for sorting)
def parse_timeslot(timeslot_str):
    try:
        # Extract the start time from the timeslot string
        # Format is typically "(dd/mm, HH:MM - HH:MM)"
        time_part = timeslot_str.strip('()')
        date_str, time_range = time_part.split(', ')
        start_time_str = time_range.split(' - ')[0]
        
        # Parse into a datetime object
        day, month = date_str.split('/')
        hour, minute = start_time_str.split(':')
        
        # Create a datetime using the current year (since it's not in the string)
        current_year = datetime.now().year
        dt = datetime(current_year, int(month), int(day), int(hour), int(minute))
        return dt
    except Exception as e:
        st.session_state.logs.append(f"Error parsing timeslot {timeslot_str}: {e}")
        return datetime.max  # Return far future date if parsing fails

def recalculate_route_around_resolved(open_cases, resolved_cases, current_position=None):
    """
    Recalculate the optimal route for open cases around fixed resolved cases.
    
    Parameters:
    - open_cases: List of tuples (coordinate, postal) for open cases
    - resolved_cases: List of tuples (start_dt, end_dt, location) for resolved cases
    - current_position: Current position to start the route from (optional)
    
    Returns:
    - A new ordered list of tuples (coordinate, postal) for open cases
    """
    if not open_cases:
        return []
    
    if not resolved_cases:
        # If no resolved cases, use regular shortest path
        start_coord = current_position if current_position else open_cases[0][0]
        coords = [c[0] for c in open_cases]
        route_indices = []
        
        # Find the starting index
        if current_position:
            start_idx = 0  # Start with the first case
        else:
            start_idx = 0  # Default to first case
            
        route_indices.append(start_idx)
        remaining_indices = set(range(len(open_cases))) - {start_idx}
        current_coord = coords[start_idx]
        
        # Find nearest neighbors
        while remaining_indices:
            nearest_idx = min(remaining_indices, key=lambda i: geodesic(current_coord, coords[i]).km)
            route_indices.append(nearest_idx)
            remaining_indices.remove(nearest_idx)
            current_coord = coords[nearest_idx]
            
        # Return reordered open cases
        return [open_cases[i] for i in route_indices]
    
    # Sort resolved cases by start time
    sorted_resolved = sorted(resolved_cases, key=lambda x: x[0])
    
    # Reorganize the route to optimize around resolved cases
    result_route = []
    current_cases = open_cases.copy()
    current_pos = current_position if current_position else (
        current_cases[0][0] if current_cases else None
    )
    
    for start_dt, end_dt, location in sorted_resolved:
        if not current_cases:
            break
            
        # Find cases that can be done before this resolved case
        # Calculate cases that can fit before the current resolved case
        available_cases = []
        remaining_cases = []
        
        # First identify which cases can be done before the resolved case
        for coord, postal in current_cases:
            # Skip if we don't have a current position yet
            if current_pos is None:
                current_pos = coord
                continue
                
            # Calculate time needed to get from current position to this case
            travel_time_min = calculate_travel_time_km(current_pos, coord)
            travel_time = timedelta(minutes=math.ceil(travel_time_min / 30) * 30)
            
            # Assume 1 hour for each case (this is simplified, in reality would use inspection_time)
            case_time = timedelta(hours=1)
            
            # Check if we can finish this case and get to the resolved case on time
            travel_to_resolved_min = calculate_travel_time_km(coord, location)
            travel_to_resolved = timedelta(minutes=math.ceil(travel_to_resolved_min / 30) * 30)
            
            # If we can reach this case from current position, complete it, 
            # and reach the resolved case before its start time
            if datetime.now() + travel_time + case_time + travel_to_resolved <= start_dt:
                available_cases.append((coord, postal))
            else:
                remaining_cases.append((coord, postal))
        
        if available_cases:
            # Find the shortest path through these available cases
            available_coords = [c[0] for c in available_cases]
            start_coord = current_pos
            
            # Create a mapping from coordinates back to cases
            coord_to_case = {c[0]: c for c in available_cases}
            
            # Find shortest path for available cases
            path_coords = find_shortest_path(available_coords, start_coord)
            path = [coord_to_case[c] for c in path_coords if c in coord_to_case]
            
            # Add these cases to the result route
            result_route.extend(path)
            
            # Update current position to the last case in the path
            if path:
                current_pos = path[-1][0]
        
        # Update current position to this resolved case's location
        current_pos = location
        
        # Update current cases to remaining cases
        current_cases = remaining_cases
    
    # Handle any remaining cases after all resolved cases
    if current_cases:
        remaining_coords = [c[0] for c in current_cases]
        if current_pos and remaining_coords:
            path_coords = find_shortest_path(remaining_coords, current_pos)
            
            # Create a mapping from coordinates back to cases
            coord_to_case = {c[0]: c for c in current_cases}
            
            # Add the remaining cases in shortest path order
            for coord in path_coords:
                if coord in coord_to_case:
                    result_route.append(coord_to_case[coord])
    
    return result_route
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

if blocked_times:
    st.session_state.blocked_slots = parse_blocked_times(blocked_times)
if holidays:
    st.session_state.holiday_set = parse_holidays(holidays_str)


# Create the base map
if st.session_state.route_data:
    # Show the route map if we have calculated a route (for open cases)
    route_with_timeslot = st.session_state.route_data
    m = folium.Map(location=route_with_timeslot[0][0], zoom_start=13)
    
    # Draw the open cases route
    for i, (coord, postal, timeslot) in enumerate(route_with_timeslot):
        # Use green for starting point, red for end point, blue for intermediate points
        color = "purple" if i == 0 else ("orange" if i == len(route_with_timeslot) - 1 else "blue")
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


# Add markers for resolved_cases with a different icon or color
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
    # Reorder open_cases based on the route order
    ordered_open_cases = [coord for coord, _, _ in route_with_timeslot]
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
        # Use datetime.strptime().time() directly to avoid potential name collision
        start_time_obj = datetime.strptime(start_time_str, "%H:%M").time()
        end_time_obj = datetime.strptime(end_time_str, "%H:%M").time()
        start_dt = datetime.combine(slot_date, start_time_obj)
        end_dt = datetime.combine(slot_date, end_time_obj)
        parsed_resolved_cases.append((start_dt, end_dt, loc))
        st.session_state.logs.append(f"Parsed resolved slot: {slot_str} -> {start_dt} to {end_dt}, Location: {loc}")
    except ValueError as e:
        st.session_state.logs.append(f"Error parsing resolved slot '{case['timeslot']}': {e}")


# Button actions
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🧮 Calculate Route"):
        if len(st.session_state.open_cases) < 2:
            st.warning("Add at least 2 open_cases.")
        else:
            startCoords = st.session_state.open_cases[0]
            
            # Get inspection time inputs (ensure they retain the previous values)
            inspection_times = []
            for i in range(len(st.session_state.open_cases)):
                inspection_time = st.session_state.inspection_times.get(i, 1)  # Default to 1 if not set
                inspection_times.append(inspection_time)
            
            # Check if we have resolved cases to consider
            if parsed_resolved_cases:
                st.session_state.logs.append(f"Calculating route with {len(parsed_resolved_cases)} resolved cases")
                
                # Create the open cases tuples (coordinate, postal)
                open_cases_with_postal = [(coord, f"OC{i+1:03}") for i, coord in enumerate(st.session_state.open_cases)]
                
                # Recalculate route considering resolved cases
                optimized_route = recalculate_route_around_resolved(
                    open_cases=open_cases_with_postal,
                    resolved_cases=parsed_resolved_cases,
                    current_position=startCoords
                )
                
                route_with_postal = optimized_route
                
                # Create a mapping from coordinates to inspection times
                coord_to_time = {}
                for i, coord in enumerate(st.session_state.open_cases):
                    coord_to_time[coord] = inspection_times[i]
                
                # Create a new list of inspection times in the same order as the route
                reordered_inspection_times = []
                for coord, _ in route_with_postal:
                    # Use the inspection time from the original order
                    inspection_time = 1  # Default
                    
                    # Find the inspection time for this coordinate
                    for orig_coord, time in coord_to_time.items():
                        if abs(orig_coord[0] - coord[0]) < 0.0001 and abs(orig_coord[1] - coord[1]) < 0.0001:
                            inspection_time = time
                            break
                    
                    reordered_inspection_times.append(inspection_time)
            else:
                st.session_state.logs.append("No resolved cases, using regular shortest path")
                shortest_route = find_shortest_path(st.session_state.open_cases, startCoords)
                route_with_postal = [(coord, f"OC{i+1:03}") for i, coord in enumerate(shortest_route)]
                reordered_inspection_times = inspection_times
            
            # When calling assign_timeslots_stable_with_travel_time, add resolved_cases parameter:
            route_with_timeslot = assign_timeslots_stable_with_travel_time(
                route_with_postal=route_with_postal,
                inspection_times=reordered_inspection_times,
                resolved_cases=parsed_resolved_cases,
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
#endregion