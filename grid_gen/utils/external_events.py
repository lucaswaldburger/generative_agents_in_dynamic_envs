
def get_external_events_for_t(t):
    if t == 0:
        return "humans receive an alert text: there is a fire, but no need to evacuate yet"
    if t == 10:
        return "the fire alarm sounds loudly, evacuation is now required. Isabella sees smoke outside of the building."
    return None