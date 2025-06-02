import marimo

__generated_with = "0.11.18"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(cal, plot_calendar):

    # Plot calendar
    fig = plot_calendar(cal)
    return (fig,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _(pd):
    from dataclasses import dataclass, field
    from datetime import datetime, timedelta
    from typing import Optional, Union, List
    import uuid
    import copy


    @dataclass
    class Event:
        name: str
        start: datetime
        end: Union[datetime, timedelta]
        id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Auto‑generated unique ID
        group: Optional[str]        = None    # Lane or category for plotting
        resource: Optional[str]     = None    # Color‑coding or secondary grouping
        linkGroup: Optional[str]    = None    # Shared ID to link multiple events
        description: Optional[str]  = None    # Will store duration in hours

        def __post_init__(self):
            # 1) Normalize end if it was provided as a duration
            if isinstance(self.end, timedelta):
                self.end = self.start + self.end  

            # 2) Default linkGroup to group if not explicitly set
            if self.linkGroup is None:
                self.linkGroup = self.group  

            # 3) Compute duration in hours and store as description
            total_seconds = (self.end - self.start).total_seconds()  
            duration_hours = total_seconds / 3600                      
            # Format: e.g. "8.0h" or round as needed
            self.description = f"duration: {duration_hours:.2f}h"               


    def events_to_dataframe(events: List[Event]) -> pd.DataFrame:
        """Convert a list of Event objects into a DataFrame for plotting."""
        return pd.DataFrame([{
            "id":          e.id,
            "name":        e.name,
            "start":       e.start,
            "end":         e.end,
            "group":       e.group,
            "resource":    e.resource,
            "description": e.description
        } for e in events])

    class Calendar:
        def __init__(self, events: List[Event]):
            # Work with a deep copy so original list isn’t mutated
            self.events: List[Event] = copy.deepcopy(events)

        def update_event(self,
                         event_id: str,
                         new_start: Optional[datetime]  = None,
                         new_end:   Optional[Union[datetime, timedelta]] = None):
            # Locate the event to update
            primary = next(e for e in self.events if e.id == event_id)

            # Handle duration-style new_end automatically
            if new_end is not None and isinstance(new_end, timedelta):
                new_end = (new_start or primary.start) + new_end

            # Compute how much to shift start/end
            delta_start = ((new_start - primary.start).total_seconds()
                           if new_start else 0)
            delta_end   = ((new_end   - primary.end).total_seconds()
                           if new_end   else 0)

            # Find all events sharing its linkGroup (or just the one)
            if primary.linkGroup:
                targets = [e for e in self.events
                           if e.linkGroup == primary.linkGroup]
            else:
                targets = [primary]

            # Apply the same delta to each linked event
            for e in targets:
                e.start = e.start + timedelta(seconds=delta_start)
                e.end   = e.end   + timedelta(seconds=delta_end)

    return (
        Calendar,
        Event,
        List,
        Optional,
        Union,
        copy,
        dataclass,
        datetime,
        events_to_dataframe,
        field,
        timedelta,
        uuid,
    )


@app.cell
def _(Calendar, events_to_dataframe):
    import plotly.express as px

    def plot_calendar(calendar: Calendar, show: bool = True):
        df = events_to_dataframe(calendar.events)
        fig = px.timeline(
            df,
            x_start="start",
            x_end="end",
            y="group",              # Grouped lanes on y-axis
            color="resource",       # Optional color-coding
            hover_data=["name", "description"]
        )
        fig.update_yaxes(autorange="reversed")  # Keep top-down order :contentReference[oaicite:6]{index=6}
        # fig.update_xaxes(
        #     tickformat="%H:%M",     # Hour:Minute format
        #     dtick=3600000           # One-hour intervals (ms) :contentReference[oaicite:7]{index=7}
        # )
        fig.update_layout(title="Linked Event Schedule", xaxis_title="Time", yaxis_title="Group")
        if show:
            fig.show()
        return fig
    return plot_calendar, px


@app.cell
def _(Calendar, Event, datetime, timedelta):
    # events = [
    #     Event("1", "Opening Ceremony", datetime(2025,5,12,9,0),  datetime(2025,5,12,10,0), resource="Main Hall"),
    #     Event("2", "Keynote",          datetime(2025,5,12,10,30),datetime(2025,5,18,11,30),resource="Main Hall"),
    #     Event("3", "Workshop A",       datetime(2025,5,12,11,45),datetime(2025,5,12,13,15),resource="Room 101"),
    #     Event("4", "Lunch Break",      datetime(2025,5,12,13,30),datetime(2025,5,12,14,30)),
    # ]

    # fig = plot_events(events)


    # Initial events with a shared linkGroup "Team"

    dissection_time = datetime(2025,6,5,9,0)

    events = [
        # HH 10
        Event("Incub to hh8",   dissection_time - timedelta(hours=9) - timedelta(hours=2) - timedelta(hours=30), timedelta(hours=30), group="HH10", resource="Blue"),
        Event("Harvest & Epor",   dissection_time - timedelta(hours=9) - timedelta(hours=2),  timedelta(hours=2), group="HH10", resource="Red", linkGroup="HH10"),
        Event("Epor Incub",      dissection_time - timedelta(hours=9), timedelta(hours=9), group="HH10", resource="Blue", linkGroup="HH10"),
        # HH 15
        Event("Incub to hh11",   dissection_time - timedelta(hours=17) - timedelta(hours=2) - timedelta(hours=42),  timedelta(hours=42), group="HH15", resource="Blue", linkGroup="HH15"),
        Event("Harvest & Epor",   dissection_time - timedelta(hours=17) - timedelta(hours=2),  timedelta(hours=2), group="HH15", resource="Red", linkGroup="HH15"),
        Event("Epor Incub",      dissection_time - timedelta(hours=17), timedelta(hours=17), group="HH15", resource="Blue", linkGroup="HH15"),
        # HH 18
        Event("Incub to hh11",   dissection_time - timedelta(hours=34) - timedelta(hours=2) - timedelta(hours=42),  timedelta(hours=42), group="HH18", resource="Blue", linkGroup="HH18"),
        Event("Harvest & Epor",   dissection_time - timedelta(hours=34) - timedelta(hours=2),  timedelta(hours=2), group="HH18", resource="Red", linkGroup="HH18"),
        Event("Epor Incub",      dissection_time - timedelta(hours=34), timedelta(hours=34),group="HH18", resource="Blue", linkGroup="HH18"),
        # Dissect & Dissociate
        Event("Dissect",      dissection_time, timedelta(minutes=45) ,group="HH10", resource="Red"),
        Event("Dissociate",   datetime(2025,6,5,10,30),  timedelta(hours=1.5), group="HH10", resource="Red"),
    ]

    cal = Calendar(events)
    return cal, dissection_time, events


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
