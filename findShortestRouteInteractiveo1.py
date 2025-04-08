import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from geopy.distance import geodesic
from datetime import datetime, timedelta, time
import math

# Add this at the top with other imports
from streamlit.components.v1 import html

# region Logic
def parse_blocked_times(blocked_times_str):
    """
    Parse a string of blocked times into a list of (start_datetime, end_datetime) tuples.
    Example input: "26/04/2025 9:30 - 10:30, 30/04/2025 13:00 - 16:30"
    """
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
    """
    Check if the given time window [current_time, current_time+duration) conflicts
    with any interval in blocked_slots.
    """
    slot_end = current_time + duration
    for blocked_start, blocked_end in blocked_slots:
        # Compare date & time, blocking only if they overlap on the same day
        if current_time.date() == blocked_start.date():
            # Overlap occurs if not ( slot_end <= blocked_start or current_time >= blocked_end )
            if not (slot_end <= blocked_start or current_time >= blocked_end):
                return True
    return False

def find_shortest_path(coords, startCoords):
    """
    Simple nearest-neighbor approach to order the open cases by minimal travel distance.
    Returns a route (list of coordinates).
    """
    current_coord = startCoords
    route = [current_coord]
    remaining = set(coords) - {startCoords}
    while remaining:
        nearest_coord = min(remaining, key=lambda c: geodesic(current_coord, c).km)
        route.append(nearest_coord)
        remaining.remove(nearest_coord)
        current_coord = nearest_coord
    return route

def calculate_travel_time_km(coord1, coord2):
    """
    Returns travel time in minutes based on distance between coords.
    For demonstration: 10 minutes per km.
    """
    return geodesic(coord1, coord2).km * 10  # 10 min/km

def parse_holidays(holidays_str):
    """
    Parse a string of holidays (comma-separated "dd/mm/YYYY") into a set of date objects.
    """
    if not holidays_str:
        return set()
    dates = [d.strip() for d in holidays_str.split(',')]
    return set(datetime.strptime(d, '%d/%m/%Y').date() for d in dates)

def add_working_days(start_date, days, holidays_set):
    """
    Add a given number of working days to a date, skipping weekends and holidays.
    """
    # Move start_date forward if it falls on a weekend/holiday
    while start_date.weekday() >= 5 or start_date in holidays_set:
        start_date += timedelta(days=1)
    
    current = start_date
    while days > 0:
        current += timedelta(days=1)
        if current.weekday() < 5 and current not in holidays_set:
            days -= 1
    return current

def assign_timeslots_stable_with_travel_time(
    route_with_postal,
    inspection_times,
    resolved_cases=None,    # <--- NEW: We pass resolved cases in
    blocked_slots=None,
    holidays_set=None,
    current_date=datetime.now().date(),
    start_hour=9,
    end_hour=18,
    max_distance_km=1,
    max_cases_per_slot=3
):
    """
    Assign timeslots for open-case inspections along a route, factoring in:
      - Travel time between cases
      - Resolved cases (fixed times) - we treat them as 'blocked' + reinsert into final schedule
      - Blocked time slots
      - Start from next 2 working days
      - Up to 3 short cases (1 hr each) can share a single timeslot if they're within 1 km
      - One long case (>1 hr) per timeslot
      - Skip weekends/holidays

    Parameters:
    - route_with_postal: [(coord, postal_code), ...] for open cases in traveling order.
    - inspection_times: list of durations (in hours) for each open case in the same order as route_with_postal.
    - resolved_cases: list of (start_dt, end_dt, coord), these are fixed time visits (cannot be changed).
    - blocked_slots: list of (start_dt, end_dt) to avoid for open-case assignment.
    - holidays_set: set of holiday dates to skip.
    - current_date: scheduling starts from 'current_date' + 2 working days.
    - start_hour, end_hour: daily working window, e.g. 9-18.
    - max_distance_km: threshold to group short open-cases in the same timeslot if within 1 km.
    - max_cases_per_slot: max short open-cases to group in 1 timeslot.

    Returns:
    - final_schedule: a list of (coord, postal, timeslot_str, is_resolved) in chronological order.
    """
    st.session_state.logs.append(f"[assign_timeslots] route_with_postal={route_with_postal}")
    st.session_state.logs.append(f"[assign_timeslots] resolved_cases={resolved_cases}")
    if blocked_slots is None:
        blocked_slots = []
    if resolved_cases is None:
        resolved_cases = []
    if holidays_set is None:
        holidays_set = set()

    # 1) Convert resolved_cases into "blocked" intervals so open cases won't conflict.
    #    We'll store them separately for final re-merging.
    #    resolved_cases: list of (start_dt, end_dt, loc)
    #    We'll add to blocked_slots so open-case scheduling avoids them.
    for r_start, r_end, _ in resolved_cases:
        blocked_slots.append((r_start, r_end))

    # 2) Our existing scheduling logic for open cases only (unchanged)
    #    Start from next 2 working days
    start_date = add_working_days(current_date, 2, holidays_set)
    base_time = datetime.combine(start_date, time(start_hour, 0))
    current_time = base_time

    timeslots = []
    group = []
    group_time = timedelta(hours=1)  # short-case group slot = 1 hour
    prev_group_end_time = base_time
    prev_group_last_coord = route_with_postal[0][0] if route_with_postal else None

    def move_to_next_day(ct):
        st.session_state.logs.append(f"[move_to_next_day] from={ct}")
        nxt = ct + timedelta(days=1)
        while nxt.weekday() >= 5 or nxt.date() in holidays_set:
            nxt += timedelta(days=1)
        return nxt.replace(hour=start_hour, minute=0, second=0, microsecond=0)

    def exceeds_end_hour(start_t, dur):
        end_t = start_t + dur
        # If end_time is strictly beyond end_hour => True
        if end_t.hour > end_hour or (end_t.hour == end_hour and end_t.minute > 0):
            return True
        return False

    def find_next_available_time(ct, dur, blocked, end_h):
        """
        Step in 30-min increments until no conflict with blocked slots
        and not exceeding end-hour.
        """
        while is_time_blocked(ct, dur, blocked) or exceeds_end_hour(ct, dur):
            ct += timedelta(minutes=30)
            if ct.hour >= end_h or exceeds_end_hour(ct, dur):
                ct = move_to_next_day(ct)
        return ct

    # Iterate over open cases in route_with_postal
    for i, (coord, postal) in enumerate(route_with_postal):
        inspection_time = inspection_times[i]
        case = (coord, postal, inspection_time)

        # If case is "long" (>1 hr), assign singly
        if inspection_time > 1:
            # If there's a partially built group, finalize it
            if group:
                current_time = find_next_available_time(current_time, group_time, blocked_slots, end_hour)
                if exceeds_end_hour(current_time, group_time):
                    current_time = move_to_next_day(current_time)
                group_end = current_time + group_time
                timeslots.append((group, current_time, group_end.strftime("%H:%M")))
                prev_group_end_time = group_end
                prev_group_last_coord = group[-1][0] if group else prev_group_last_coord
                current_time = group_end
                group = []

            # Travel from previous group's last coord to this new case
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
            timeslots.append(([case], start, end.strftime("%H:%M")))
            prev_group_end_time = end
            prev_group_last_coord = coord
            current_time = end

        else:
            # short case logic
            if not group:
                # If starting a new group, factor in travel from prev group
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
                # If next case is within 1 km and group not full, group it
                if distance_km < max_distance_km and len(group) < max_cases_per_slot:
                    group.append(case)
                else:
                    # finalize the old group
                    current_time = find_next_available_time(current_time, group_time, blocked_slots, end_hour)
                    if exceeds_end_hour(current_time, group_time):
                        current_time = move_to_next_day(current_time)
                    group_end = current_time + group_time
                    timeslots.append((group, current_time, group_end.strftime("%H:%M")))
                    prev_group_end_time = group_end
                    prev_group_last_coord = group[-1][0]
                    current_time = group_end
                    group = [case]

                    # Then handle travel to new group's first case
                    if prev_group_end_time is not None:
                        travel_time_min = calculate_travel_time_km(prev_group_last_coord, coord)
                        rounded_travel_time = timedelta(minutes=math.ceil(travel_time_min / 30) * 30)
                        new_start_time = prev_group_end_time + rounded_travel_time
                        if exceeds_end_hour(new_start_time, timedelta()):
                            new_start_time = move_to_next_day(new_start_time)
                        current_time = new_start_time
                    if exceeds_end_hour(current_time, group_time):
                        current_time = move_to_next_day(current_time)

    # If there's a remaining group, finalize it
    if group:
        current_time = find_next_available_time(current_time, group_time, blocked_slots, end_hour)
        if exceeds_end_hour(current_time, group_time):
            current_time = move_to_next_day(current_time)
        group_end = current_time + group_time
        timeslots.append((group, current_time, group_end.strftime("%H:%M")))

    # 3) Build a preliminary open-case schedule
    #    timeslots = [ ( [ (coord, postal, inspTime), ... ], start_dt, end_time_str ), ... ]
    #    We'll create result_timeslots for open cases only, but in chronological order
    result_open_timeslots = []
    for group_list, start_dt, end_time_str in timeslots:
        slot_str = f"({start_dt.strftime('%d/%m, %H:%M')} - {end_time_str})"
        for (coord, postal, _) in group_list:
            result_open_timeslots.append((coord, postal, slot_str, False))  # is_resolved=False

    # 4) Combine with resolved cases → final schedule
    #    resolved_cases: list of (r_start, r_end, r_coord)
    #    We'll build a single timeline. For open-cases, parse the slot_str to figure out start time.
    #    For resolved cases, we have direct r_start, r_end.

    def parse_open_timeslot_str(timeslot_str):
        # Format: "(dd/mm, HH:MM - HH:MM)"
        # e.g.: "(26/04, 09:00 - 10:00)"
        # parse date & time
        # We'll assume day/month doesn't contain year, so we assume the year is the current for scheduling:
        # But to keep consistent with a single approach, let's use the current year from start_dt or so.
        inside = timeslot_str.strip("()")
        date_part, times_part = inside.split(',', 1)  # "26/04" and " HH:MM - HH:MM"
        date_part = date_part.strip()
        times_part = times_part.strip()
        # date_part might be "26/04", we will guess the current scheduling year:
        # Or we can try to detect. We'll do a naive approach:
        # We'll guess we started scheduling in year = start_date.year
        # (the code doesn't do cross-year scheduling in examples, so it's okay for demo).
        scheduling_year = start_date.year
        dd, mm = map(int, date_part.split('/'))
        # times_part = "09:00 - 10:00"
        start_t_str, end_t_str = times_part.split('-')
        start_t_str = start_t_str.strip()
        end_t_str = end_t_str.strip()
        start_h, start_m = map(int, start_t_str.split(':'))
        end_h, end_m = map(int, end_t_str.split(':'))
        start_dt_ = datetime(scheduling_year, mm, dd, start_h, start_m)
        end_dt_ = datetime(scheduling_year, mm, dd, end_h, end_m)
        return start_dt_, end_dt_

    final_list = []
    # Convert open-case schedules to a structure with actual start_dt
    for coord, postal, slot_str, is_resolved in result_open_timeslots:
        start_dt_, end_dt_ = parse_open_timeslot_str(slot_str)
        final_list.append((start_dt_, end_dt_, coord, postal, is_resolved))

    # Add the resolved cases in the same structure
    for (r_start, r_end, r_coord) in resolved_cases:
        # We don't have a "postal" string for these, so let's label them "R-Resolved"
        final_list.append((r_start, r_end, r_coord, "RC_Resolved", True))

    # 5) Sort everything by start_dt
    final_list.sort(key=lambda x: x[0])  # sort by start_dt

    # 6) Build final schedule: (coord, postal, timeslot_str, is_resolved) in chronological order
    final_schedule = []
    for (start_dt_, end_dt_, coord, postal, is_resolved) in final_list:
        slot_str = f"({start_dt_.strftime('%d/%m, %H:%M')} - {end_dt_.strftime('%H:%M')})"
        final_schedule.append((coord, postal, slot_str, is_resolved))

    return final_schedule
# endregion

# region Streamlit
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
    st.session_state.pending_resolved = None
if "resolved_case_counter" not in st.session_state:
    st.session_state.resolved_case_counter = 0

# Show debug logs
def show_logs():
    with st.expander("🐛 Debug Logs", expanded=False):
        if st.button("Clear Logs"):
            st.session_state.logs = []
        log_text = "\n".join(st.session_state.logs)
        st.code(log_text, language="plaintext")

st.set_page_config(layout="centered")
st.title("Appoinment Scheduling Simulation")

# Sidebar
st.sidebar.header("Additional Settings")

mode = st.sidebar.radio("Select Mode", options=["Add Open Case", "Add Resolved Case"])
st.session_state.add_mode = "open" if mode == "Add Open Case" else "resolved"

blocked_times = st.sidebar.text_input(
    "Enter blocked time slots (format: '26/04/2025 9:30 - 10:30, 30/04/2025 13:00 - 16:30')",
    "07/04/2025 9:30 - 10:30, 08/04/2025 8:00 - 18:00, 30/04/2025 13:00 - 16:30",
    help="Input Lunch Breaks, Leaves, Holiday, Appointment. If whole day is blocked, e.g. 'dd/mm/yyyy 9:00 - 18:00'."
         " Use commas to separate multiple time slots. 24-hour format."
)
holidays = st.sidebar.multiselect(
    "Holidays",
    options=[(datetime.now() + timedelta(days=i)).strftime("%d/%m/%Y") for i in range(0, 365)],
    default=["08/04/2025", "09/04/2025"],
    help="Select dates where the whole day is blocked (public holidays)."
)
current_date = st.sidebar.date_input("Current Date", datetime.now().date(), format="DD/MM/YYYY")
st.session_state.current_date = current_date

st.sidebar.write("Blocked Time Slots:", blocked_times)
holidays_str = ", ".join(holidays)
st.sidebar.write("Holidays:", holidays_str)
st.sidebar.write("Current Date:", current_date.strftime("%d/%m/%Y"))

# Sidebar: resolved cases info
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

# Sidebar: open cases
st.sidebar.subheader("Open Cases")
if st.session_state.open_cases:
    open_cases_info = ""
    for i, point in enumerate(st.session_state.open_cases):
        lat, lng = point
        case_id = f"OC{i+1:03}"
        inspection_time = st.session_state.inspection_times.get(i, 1)  # default 1 hour
        timeslot = "Not scheduled yet"
        if st.session_state.route_data:
            for coord, postal, ts, is_resolved in st.session_state.route_data:
                if not is_resolved:
                    # Match by approximate coordinate
                    if (abs(coord[0] - lat) < 0.0001 and abs(coord[1] - lng) < 0.0001 and postal == case_id):
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

# Parse blocked times & holidays
if blocked_times:
    st.session_state.blocked_slots = parse_blocked_times(blocked_times)
if holidays:
    st.session_state.holiday_set = parse_holidays(holidays_str)

# Create base map
if st.session_state.route_data:
    # If we have a final route (open + resolved) in chronological order, show it
    route_with_timeslot = st.session_state.route_data
    # Start map from earliest
    m = folium.Map(location=route_with_timeslot[0][0], zoom_start=13)

    for i, (coord, postal, timeslot, is_resolved) in enumerate(route_with_timeslot):
        if is_resolved:
            # Mark resolved in green
            color = "green"
        else:
            # Use purple for the very first open, orange for the last open, blue for others
            if i == 0:
                color = "purple"
            elif i == len(route_with_timeslot) - 1:
                color = "orange"
            else:
                color = "blue"

        folium.Marker(coord, icon=folium.Icon(color=color)).add_to(m)
        folium.map.Marker(
            location=coord,
            icon=folium.DivIcon(
                icon_size=(150, 36),
                html=(
                    f'<div style="font-size: 12px; color: black; background-color: white; '
                    f'padding: 2px; border-radius: 4px; border: 1px solid grey;">'
                    f'{postal}<br>{timeslot}</div>'
                )
            )
        ).add_to(m)

    # Draw lines in order (chronological)
    for i in range(len(route_with_timeslot) - 1):
        c1, _, _, _ = route_with_timeslot[i]
        c2, _, _, _ = route_with_timeslot[i + 1]
        dist_km = geodesic(c1, c2).km
        midpoint = ((c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2)
        folium.PolyLine([c1, c2], color="blue", weight=3).add_to(m)
        folium.Marker(midpoint, icon=folium.DivIcon(
            html=f'<div style="font-size: 11px; color: black;">{dist_km:.2f} km</div>'
        )).add_to(m)

    Draw(
        export=False,
        draw_options={"polyline": False, "rectangle": False, "circle": False, "circlemarker": False, "polygon": False},
        edit_options={"edit": False, "remove": True}
    ).add_to(m)
else:
    # Default map
    m = folium.Map(location=[1.3521, 103.8198], zoom_start=12)
    Draw(
        export=False,
        draw_options={"polyline": False, "rectangle": False, "circle": False, "circlemarker": False, "polygon": False},
        edit_options={"edit": False, "remove": True}
    ).add_to(m)

# Display map + handle drawn geometry
output = st_folium(m, width=700, height=500, key="map", returned_objects=["last_active_drawing", "all_drawings"])

# Process newly drawn case
if output["last_active_drawing"]:
    coords = output["last_active_drawing"]["geometry"]["coordinates"]
    latlng = (coords[1], coords[0])
    if st.session_state.add_mode == "open":
        if latlng not in st.session_state.open_cases:
            st.session_state.open_cases.append(latlng)
            st.session_state.show_route = False  # reset route
    else:
        # "resolved" mode, store pending to ask user for timeslot
        st.session_state.pending_resolved = latlng
        st.session_state.logs.append(f"Selected Resolved Case at: {latlng}")

# If pending resolved case, show an input field to store timeslot
if st.session_state.pending_resolved:
    with st.expander("Enter Resolved Case Timeslot", expanded=True):
        timeslot_input = st.text_input(
            "Enter timeslot (e.g. '20/04/2025 09:00 - 10:00')",
            value="11/04/2025 11:00 - 12:00",
            help="Fixed timeslot for the resolved case (cannot be changed later)."
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

# Parse resolved cases into (start_dt, end_dt, loc)
parsed_resolved_cases = []
for case in st.session_state.resolved_cases:
    try:
        slot_str = case["timeslot"]
        loc = case["location"]
        date_part, times_part = slot_str.split(" ", 1)  # e.g. "11/04/2025", "11:00 - 12:00"
        start_time_str, end_time_str = times_part.split(" - ")
        slot_date = datetime.strptime(date_part, "%d/%m/%Y").date()
        start_dt = datetime.combine(slot_date, datetime.strptime(start_time_str, "%H:%M").time())
        end_dt = datetime.combine(slot_date, datetime.strptime(end_time_str, "%H:%M").time())
        parsed_resolved_cases.append((start_dt, end_dt, loc))
    except ValueError as e:
        st.session_state.logs.append(f"Error parsing resolved slot '{case['timeslot']}': {e}")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🧮 Calculate Route"):
        if len(st.session_state.open_cases) < 1 and len(st.session_state.resolved_cases) < 1:
            st.warning("Add at least one Open Case or one Resolved Case.")
        else:
            # 1) If we have multiple open cases, do a TSP among them
            #    If we have at least 2 open cases, find a route
            open_cases = st.session_state.open_cases
            if len(open_cases) >= 2:
                startCoords = open_cases[0]
                shortest_route = find_shortest_path(open_cases, startCoords)
                route_with_postal = [(coord, f"OC{i+1:03}") for i, coord in enumerate(shortest_route)]
            else:
                # only 0 or 1 open case
                route_with_postal = []
                for i, coord in enumerate(open_cases):
                    route_with_postal.append((coord, f"OC{i+1:03}"))

            # 2) Gather inspection times in the same order as route
            inspection_times_ordered = []
            for (coord, postal) in route_with_postal:
                # find the index in st.session_state.open_cases
                idx = st.session_state.open_cases.index(coord)
                insp_t = st.session_state.inspection_times.get(idx, 1)
                inspection_times_ordered.append(insp_t)

            # 3) Call the updated assignment function
            final_schedule = assign_timeslots_stable_with_travel_time(
                route_with_postal=route_with_postal,
                inspection_times=inspection_times_ordered,
                resolved_cases=parsed_resolved_cases,
                blocked_slots=st.session_state.blocked_slots,
                holidays_set=st.session_state.holiday_set,
                current_date=st.session_state.current_date
            )
            # final_schedule is a list of (coord, postal, timeslot, is_resolved) in chronological order
            st.session_state.route_data = final_schedule
            st.session_state.show_route = True
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

# Display open_cases (allows user to set inspection times and remove them)
if st.session_state.open_cases:
    st.session_state.logs.append("Current open_cases:")
    for i, point in enumerate(st.session_state.open_cases):
        lat, lng = point
        label = f"OC{i+1:03}"
        st.session_state.logs.append(f"Point {label}: {lat:.4f}, {lng:.4f}")
        inspection_time = st.number_input(
            f"Inspection Time for {label}",
            min_value=1, max_value=8,
            value=st.session_state.inspection_times.get(i, 1),
            key=f"inspection_time_{i}"
        )
        st.session_state.inspection_times[i] = inspection_time
        delete_button = st.button(f"❌ Delete {label}", key=f"delete_{i}")
        if delete_button:
            st.session_state.open_cases.pop(i)
            st.session_state.show_route = False
            st.session_state.route_data = None
            st.success(f"✅ Case {label} deleted successfully.")
            st.rerun()

# Display resolved cases with a delete button
if st.session_state.resolved_cases:
    st.subheader("Resolved Cases")
    for i, case in enumerate(st.session_state.resolved_cases):
        loc = case["location"]
        ts = case["timeslot"]
        st.write(f"Resolved Case {i+1}: {case['case_id']}, Location={loc}, Timeslot={ts}")
        delete_resolved = st.button(f"❌ Delete Resolved Case {i+1}", key=f"delete_resolved_{i}")
        if delete_resolved:
            st.session_state.resolved_cases.pop(i)
            st.success(f"Resolved Case {i+1} deleted successfully.")
            st.rerun()

show_logs()
# endregion
