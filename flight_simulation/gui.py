

"""Tkinter GUI entrypoint.

Run:
    python gui.py

If this aborts with a message like:
    macOS 15 (1507) or later required, have instead 15 (1506) !
that is a Tk/Tkinter binary compatibility issue on your machine (not a Python exception).
See README for fixes.
"""



from __future__ import annotations





def main() -> None:



    import tkinter as tk



    from src.gui_app import FlightRouteApp



    root = tk.Tk()

    FlightRouteApp(root)

    root.mainloop()





if __name__ == "__main__":

    main()

